import numpy as np
import torch
import torchvision.transforms as tvf
from PIL import Image
from PIL.ImageOps import exif_transpose

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid

try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False

ImgNorm = tvf.Compose(
    [tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def load_dust3r_model(checkpoint_path, device="cuda"):
    model = AsymmetricCroCo3DStereo.from_pretrained(checkpoint_path)
    model = model.to(device)
    model.eval()
    return model


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = Image.LANCZOS
    elif S <= long_edge_size:
        interp = Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


# Convert the input image to the format required by dust3r
def torch_images_to_dust3r_format(tensor_images, size, square_ok=False, verbose=False):
    """
    Convert a list of torch tensor images to the format required by the DUSt3R model.
    
    Args:
    - tensor_images (list of torch.Tensor): List of RGB images in torch tensor format.
    - size (int): Target size for the images.
    - square_ok (bool): Whether square images are acceptable.
    - verbose (bool): Whether to print verbose messages.

    Returns:
    - list of dict: Converted images in the required format.
    """
    imgs = []
    for idx, image in enumerate(tensor_images):
        image = image.detach().permute(1, 2, 0).cpu().numpy() * 255
        image = image.astype(np.uint8)

        img = Image.fromarray(image, "RGB")
        img = exif_transpose(img).convert("RGB")
        W1, H1 = img.size
        if size == 224:
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            img = _resize_pil_image(img, size)
        W, H = img.size
        cx, cy = W // 2, H // 2
        if size == 224:
            half = min(cx, cy)
            img = img.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not square_ok and W == H:
                halfh = 3 * halfw // 4
            img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        if verbose:
            W2, H2 = img.size
            print(
                f" - processed image {idx} with resolution "
                f"{W1}x{H1} --> {W2}x{H2}"
            )
        imgs.append(
            dict(
                img=ImgNorm(img)[None],
                true_shape=np.int32([img.size[::-1]]),
                idx=idx,
                instance=str(idx),
            )
        )

    assert imgs, "no images found"
    return imgs


# Input image pair to obtain relative pose, point cloud, and point matching correspondence
@torch.no_grad()
def get_result(
    img1,
    img2,
    model,
    device,
    batch_size=1,
    image_size=512,
    verbose=False,
):
    pairs = make_pairs(
        torch_images_to_dust3r_format(
            [img1, img2], size=image_size, square_ok=True, verbose=verbose
        ),
        scene_graph="complete",
        prefilter=None,
        symmetrize=True,
    )
    output = inference(pairs, model, device, batch_size=batch_size, verbose=verbose)
    scene = global_aligner(
        output, device=device, mode=GlobalAlignerMode.PairViewer, verbose=verbose
    )
    poses = scene.get_im_poses()
    poses_np = [pose.detach().cpu().numpy() for pose in poses]
    pts3d = scene.get_pts3d()
    pts3d_np = [pts.detach().cpu().numpy() for pts in pts3d]
    depthmaps = scene.get_depthmaps()
    depthmaps_np = [depth.detach().cpu().numpy() for depth in depthmaps]
    confidence_masks = scene.get_masks()
    masks_np = [mask.detach().cpu().numpy().astype(bool) for mask in confidence_masks]
    imgs = scene.imgs

    # Determine which frame is the reference and get relative pose
    identity_matrix = np.eye(4, 4)
    pose_diffs = [np.linalg.norm(pose - identity_matrix, ord="fro") for pose in poses_np]
    reference_idx = int(np.argmin(pose_diffs))
    if reference_idx == 0:
        trans_pose = np.linalg.inv(poses_np[1])
    else:
        trans_pose = poses_np[0]

    # point matching
    pts2d_list, pts3d_list = [], []

    for i in range(2):
        conf_i = masks_np[i]
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])
        pts3d_list.append(pts3d_np[i][conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)

    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]
    matches_3d0 = pts3d_list[0][nn2_in_P1][reciprocal_in_P2]
    matches_3d1 = pts3d_list[1][reciprocal_in_P2]

    return (
        trans_pose,
        pts3d_np,
        imgs,
        masks_np,
        reference_idx,
        matches_im0,
        matches_im1,
        matches_3d0,
        matches_3d1,
        poses_np,
        depthmaps_np,
    )

# Obtain the scale factor based on point matching correspondence and point cloud coordinates    
def get_scale(matches_im1_0, matches_im1_1, matches_im2_0, matches_im2_1, matches_3d1_0, matches_3d2_1):
    sorted_index = np.lexsort((matches_im1_1[:,0], matches_im1_1[:,1]))
    matches_im1_1s = matches_im1_1[sorted_index]

    matching_index_1 = []
    matching_index_2 = []
    k1 = 0
    k2 = 0
    
    # Extract the index of the corresponding points
    for j in range(min(np.min(matches_im1_1s[:,1]),np.min(matches_im2_0[:,1])), max(np.max(matches_im1_1s[:,1]),np.max(matches_im2_0[:,1]))+1):
        for i in range(min(np.min(matches_im1_1s[:,0]),np.min(matches_im2_0[:,0])), max(np.max(matches_im1_1s[:,0]),np.max(matches_im2_0[:,0]))+1):
            if np.array_equal(matches_im1_1s[k1], [i,j]) and np.array_equal(matches_im2_0[k2], [i,j]):
                matching_index_1.append(sorted_index[k1])
                matching_index_2.append(k2)
            if np.array_equal(matches_im1_1s[k1], [i,j]):
                k1 = k1+1
            if np.array_equal(matches_im2_0[k2], [i,j]):
                #print(i,j)
                k2 = k2+1
            if k2 == len(matches_im2_0) or k1 == len(matches_im1_1s):
                break
        if k2 == len(matches_im2_0) or k1 == len(matches_im1_1s):
            break
    
    #matches_im0 = matches_im1_0[matching_index_1]
    #matches_im2 = matches_im2_1[matching_index_2]
    #print(matches_im0.shape)
    #print(matches_im2.shape)
    #print(matches_3d1_0.shape)
    #print(matches_3d2_1.shape)
    
    pcd0 = matches_3d1_0[matching_index_1]
    pcd2 = matches_3d2_1[matching_index_2]
    
    num_matches = len(matching_index_1)
    match_idx_to_viz = range(0,num_matches)
    
    sample_matches_pcd0 = pcd0[match_idx_to_viz]
    sample_matches_pcd2 = pcd2[match_idx_to_viz]
    
    # Calculate the scale factor
    scale = []  
    eps = 1e-8
    for i in range(1, len(match_idx_to_viz)):
        diff1 = sample_matches_pcd0[i,:]-sample_matches_pcd0[0,:]
        diff1 = np.append(diff1, np.sqrt(diff1[0]**2+diff1[1]**2+diff1[2]**2))
        diff2 = sample_matches_pcd2[i,:]-sample_matches_pcd2[0,:]
        diff2 = np.append(diff2, np.sqrt(diff2[0]**2+diff2[1]**2+diff2[2]**2))
        scale.append(diff2/(diff1 + eps))

    scale_array = np.array(scale)
    scale_mean = np.mean(scale_array[:,3])
    scale_median = np.median(scale_array[:,3])

    return scale_mean, scale_median
