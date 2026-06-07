import torch


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def _dust3r_depth_prior_cfg(config):
    """Read the DUSt3R depth-prior loss settings (disabled by default)."""
    dcfg = config["Training"].get("dust3r", {}).get("depth_prior", {})
    return {
        "enabled": bool(dcfg.get("enabled", False)),
        "tracking_weight": float(dcfg.get("tracking_weight", 0.0)),
        "mapping_weight": float(dcfg.get("mapping_weight", 0.0)),
        "min_conf": float(dcfg.get("min_conf", 0.0)),
        "opacity_threshold": float(dcfg.get("opacity_threshold", 0.5)),
    }


def get_loss_depth_prior(depth, opacity, viewpoint, min_conf=0.0, opacity_threshold=0.5):
    """L1 between rendered depth and the DUSt3R depth prior stored on viewpoint.

    Mirrors the RGB-D depth term but uses DUSt3R depth (set at bootstrap/refresh
    keyframes) as the pseudo-GT. Only well-rendered, confident pixels contribute,
    so this nudges pose toward the DUSt3R geometry without overriding RGB.
    Returns None when no usable DUSt3R depth is available.
    """
    prior = getattr(viewpoint, "dust3r_depth", None)
    if prior is None:
        return None
    if not torch.is_tensor(prior):
        prior = torch.from_numpy(prior)
    prior = prior.to(dtype=depth.dtype, device=depth.device).view(*depth.shape)

    valid = torch.isfinite(prior) & (prior > 0.01)
    valid = valid & (opacity.view(*depth.shape) > opacity_threshold)
    conf = getattr(viewpoint, "dust3r_depth_conf", None)
    if conf is not None and min_conf > 0.0:
        if not torch.is_tensor(conf):
            conf = torch.from_numpy(conf)
        conf = conf.to(dtype=depth.dtype, device=depth.device).view(*depth.shape)
        valid = valid & (conf >= min_conf)
    if valid.count_nonzero() == 0:
        return None
    l1_depth = torch.abs(depth - prior)[valid]
    return l1_depth.mean()


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        loss = get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
        dcfg = _dust3r_depth_prior_cfg(config)
        if dcfg["enabled"] and dcfg["tracking_weight"] > 0.0:
            depth_loss = get_loss_depth_prior(
                depth,
                opacity,
                viewpoint,
                min_conf=dcfg["min_conf"],
                opacity_threshold=dcfg["opacity_threshold"],
            )
            if depth_loss is not None:
                loss = loss + dcfg["tracking_weight"] * depth_loss
        return loss
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    return l1.mean()


def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Training"]["monocular"]:
        loss = get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
        dcfg = _dust3r_depth_prior_cfg(config)
        if dcfg["enabled"] and dcfg["mapping_weight"] > 0.0:
            depth_loss = get_loss_depth_prior(
                depth,
                opacity,
                viewpoint,
                min_conf=dcfg["min_conf"],
                opacity_threshold=dcfg["opacity_threshold"],
            )
            if depth_loss is not None:
                loss = loss + dcfg["mapping_weight"] * depth_loss
        return loss
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)

    return l1_rgb.mean()


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]

    gt_image = viewpoint.original_image.cuda()

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*depth.shape)
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * l1_rgb.mean() + (1 - alpha) * l1_depth.mean()


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
