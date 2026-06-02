#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian_splatting.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian_splatting.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian_splatting.utils.sh_utils import RGB2SH
from gaussian_splatting.utils.system_utils import mkdir_p
from utils.logging_utils import Log


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()
        self.lifecycle_age = torch.empty(0, device="cuda").int()
        self.lifecycle_visibility = torch.empty(0, device="cuda").int()
        self.lifecycle_recent_visibility = torch.empty(0, device="cuda").int()
        self.lifecycle_bad_count = torch.empty(0, device="cuda").int()
        self.lifecycle_state = torch.empty(0, device="cuda").int()

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        self.isotropic = False

    def init_lifecycle(self, n_points=None):
        if n_points is None:
            n_points = self.get_xyz.shape[0]
        self.lifecycle_age = torch.zeros((n_points), device="cuda").int()
        self.lifecycle_visibility = torch.zeros((n_points), device="cuda").int()
        self.lifecycle_recent_visibility = torch.zeros((n_points), device="cuda").int()
        self.lifecycle_bad_count = torch.zeros((n_points), device="cuda").int()
        self.lifecycle_state = torch.zeros((n_points), device="cuda").int()

    def lifecycle_enabled(self):
        return bool(self.config["Training"].get("lifecycle", {}).get("enabled", False))

    def append_lifecycle(self, n_new):
        if not self.lifecycle_enabled():
            return
        if self.lifecycle_state.shape[0] == 0 and self.get_xyz.shape[0] > n_new:
            self.init_lifecycle(self.get_xyz.shape[0] - n_new)
        new_zeros = torch.zeros((n_new), device="cuda").int()
        self.lifecycle_age = torch.cat((self.lifecycle_age, new_zeros))
        self.lifecycle_visibility = torch.cat((self.lifecycle_visibility, new_zeros))
        self.lifecycle_recent_visibility = torch.cat(
            (self.lifecycle_recent_visibility, new_zeros)
        )
        self.lifecycle_bad_count = torch.cat((self.lifecycle_bad_count, new_zeros))
        self.lifecycle_state = torch.cat((self.lifecycle_state, new_zeros))

    def update_lifecycle(self, recent_visibility):
        if not self.lifecycle_enabled() or self.get_xyz.shape[0] == 0:
            return None

        cfg = self.config["Training"]["lifecycle"]
        if self.lifecycle_state.shape[0] != self.get_xyz.shape[0]:
            self.init_lifecycle(self.get_xyz.shape[0])

        recent_visibility = recent_visibility.to(device="cuda").int()
        self.lifecycle_age += 1
        self.lifecycle_recent_visibility = recent_visibility
        self.lifecycle_visibility += recent_visibility

        opacity = self.get_opacity.detach().squeeze()
        grads = self.xyz_gradient_accum / torch.clamp_min(self.denom, 1.0)
        grads[grads.isnan()] = 0.0
        grad_norm = torch.norm(grads, dim=-1)

        newborn_grace = int(cfg.get("newborn_grace", 3))
        stable_min_visibility = int(cfg.get("stable_min_visibility", 3))
        cold_min_age = int(cfg.get("cold_min_age", 10))
        cold_grad_threshold = float(cfg.get("cold_grad_threshold", 1e-5))
        cold_opacity_threshold = float(cfg.get("cold_opacity_threshold", 0.6))
        bad_opacity_threshold = float(cfg.get("bad_opacity_threshold", 0.05))
        bad_min_visibility = int(cfg.get("bad_min_visibility", 1))
        bad_use_recent_visibility = bool(
            cfg.get("bad_use_recent_visibility", False)
        )
        bad_patience = int(cfg.get("bad_patience", 3))

        past_grace = self.lifecycle_age > newborn_grace
        bad_candidate = past_grace & (opacity < bad_opacity_threshold)
        if bad_use_recent_visibility and bad_min_visibility > 0:
            low_recent_visibility = (
                self.lifecycle_recent_visibility < bad_min_visibility
            )
            bad_candidate = torch.logical_or(
                bad_candidate, past_grace & low_recent_visibility
            )
        self.lifecycle_bad_count = torch.where(
            bad_candidate,
            self.lifecycle_bad_count + 1,
            torch.zeros_like(self.lifecycle_bad_count),
        )

        bad = self.lifecycle_bad_count >= bad_patience
        cold = (
            past_grace
            & (self.lifecycle_age >= cold_min_age)
            & (self.lifecycle_recent_visibility >= stable_min_visibility)
            & (opacity >= cold_opacity_threshold)
            & (grad_norm < cold_grad_threshold)
            & ~bad
        )
        stable = (
            past_grace
            & (self.lifecycle_visibility >= stable_min_visibility)
            & ~cold
            & ~bad
        )

        state = torch.zeros_like(self.lifecycle_state)
        state[stable] = 1
        state[cold] = 2
        state[bad] = 3
        self.lifecycle_state = state
        return self.lifecycle_counts()

    def lifecycle_counts(self):
        if self.lifecycle_state.shape[0] == 0:
            return {"newborn": 0, "stable": 0, "cold": 0, "bad": 0}
        return {
            "newborn": int((self.lifecycle_state == 0).sum().item()),
            "stable": int((self.lifecycle_state == 1).sum().item()),
            "cold": int((self.lifecycle_state == 2).sum().item()),
            "bad": int((self.lifecycle_state == 3).sum().item()),
        }

    def cold_mask(self):
        if not self.lifecycle_enabled() or self.lifecycle_state.shape[0] == 0:
            return None
        return self.lifecycle_state == 2

    def bad_mask(self):
        if not self.lifecycle_enabled() or self.lifecycle_state.shape[0] == 0:
            return None
        return self.lifecycle_state == 3

    def freeze_cold_gradients(self):
        cold_mask = self.cold_mask()
        cfg = self.config["Training"].get("lifecycle", {})
        if cold_mask is None or not bool(cfg.get("freeze_cold", True)):
            return
        if not cold_mask.any():
            return

        for param in [
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._opacity,
            self._scaling,
            self._rotation,
        ]:
            if param.grad is not None:
                param.grad[cold_mask] = 0

        for group in self.optimizer.param_groups:
            param = group["params"][0]
            state = self.optimizer.state.get(param, None)
            if state is None:
                continue
            if "exp_avg" in state:
                state["exp_avg"][cold_mask] = 0
            if "exp_avg_sq" in state:
                state["exp_avg_sq"][cold_mask] = 0

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None):
        cam = cam_info
        image_ab = (torch.exp(cam.exposure_a)) * cam.original_image + cam.exposure_b
        image_ab = torch.clamp(image_ab, 0.0, 1.0)
        rgb_raw = (image_ab * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()

        if depthmap is not None:
            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depthmap.astype(np.float32))
        else:
            depth_raw = cam.depth
            if depth_raw is None:
                depth_raw = np.empty((cam.image_height, cam.image_width))

            if self.config["Dataset"]["sensor_type"] == "monocular":
                depth_raw = (
                    np.ones_like(depth_raw)
                    + (np.random.randn(depth_raw.shape[0], depth_raw.shape[1]) - 0.5)
                    * 0.05
                ) * scale

            rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
            depth = o3d.geometry.Image(depth_raw.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        if new_xyz.shape[0] == 0:
            raise ValueError("Image/depth point cloud is empty after downsampling")

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(new_rgb).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(fused_point_cloud),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def create_pcd_from_dust3r(
        self,
        pts3d,
        imgs,
        transform,
        scale=1.0,
        mask=None,
        init=False,
        pointmap_indices=None,
        alignment_transform=None,
    ):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        dust3r_config = self.config["Training"].get("dust3r", {})
        dust3r_init_config = dust3r_config.get("init", {})
        if init:
            downsample_factor = int(
                dust3r_init_config.get("pcd_downsample", downsample_factor)
            )
            point_size *= float(dust3r_init_config.get("point_size_scale", 1.0))
        depth_min = float(dust3r_config.get("depth_min", 0.05))
        depth_max = float(dust3r_config.get("depth_max", 20.0))
        max_radius = float(dust3r_config.get("max_point_radius", 30.0))
        outlier_config = dust3r_config.get("outlier_filter", {})
        outlier_enabled = bool(outlier_config.get("enabled", True))
        outlier_nb_neighbors = int(outlier_config.get("nb_neighbors", 20))
        outlier_std_ratio = float(outlier_config.get("std_ratio", 2.0))
        scale = float(scale)
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            scale = 1.0
        filter_scale = abs(scale)
        filter_depth_min = depth_min * filter_scale
        filter_depth_max = depth_max * filter_scale
        filter_max_radius = max_radius * filter_scale

        pts3d = [
            p.detach().cpu().numpy() if hasattr(p, "detach") else np.asarray(p)
            for p in pts3d
        ]
        imgs = [np.asarray(img) for img in imgs]
        if pointmap_indices is None:
            pointmap_indices = range(len(pts3d))
        if mask is not None:
            mask = [
                None if m is None else np.asarray(m).astype(bool)
                for m in mask
            ]

        pts_chunks = []
        color_chunks = []
        stats = []
        for idx in pointmap_indices:
            points = pts3d[idx]
            colors = imgs[idx]
            finite = np.isfinite(points).all(axis=-1)
            z_min = float(np.nanmin(points[..., 2])) if np.any(finite) else float("nan")
            z_max = float(np.nanmax(points[..., 2])) if np.any(finite) else float("nan")
            radius = np.linalg.norm(points, axis=-1)
            valid = finite
            valid = np.logical_and(valid, points[..., 2] > filter_depth_min)
            valid = np.logical_and(valid, points[..., 2] < filter_depth_max)
            valid = np.logical_and(valid, radius < filter_max_radius)
            if mask is not None and mask[idx] is not None:
                valid = np.logical_and(valid, mask[idx])
            stats.append(
                (
                    idx,
                    int(points.shape[0] * points.shape[1]),
                    int(np.count_nonzero(finite)),
                    int(np.count_nonzero(valid)),
                    z_min,
                    z_max,
                )
            )
            if not np.any(valid):
                continue
            pts_chunks.append(points[valid].reshape(-1, 3))
            color_chunks.append(colors[valid].reshape(-1, 3))

        if not pts_chunks:
            stat_msg = ", ".join(
                f"idx {idx}: valid {valid}/{total}, finite {finite}, z [{z_min:.3f}, {z_max:.3f}]"
                for idx, total, finite, valid, z_min, z_max in stats
            )
            raise ValueError(
                "DUSt3R did not produce any valid points for Gaussian init "
                f"(metric depth range [{depth_min}, {depth_max}], "
                f"metric max radius {max_radius}, scale divisor {scale}; "
                f"DUSt3R filter depth range [{filter_depth_min}, {filter_depth_max}], "
                f"filter max radius {filter_max_radius}; "
                f"{stat_msg})"
            )

        points = np.concatenate(pts_chunks, axis=0)
        colors = np.concatenate(color_chunks, axis=0).astype(np.float32)
        if colors.max() > 1.0:
            colors = colors / 255.0
        colors = np.clip(colors, 0.0, 1.0)

        points = points * (1.0 / scale)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        if hasattr(transform, "detach"):
            transform = transform.detach().cpu().numpy()
        pcd.transform(np.asarray(transform, dtype=np.float64))
        if alignment_transform is not None:
            if hasattr(alignment_transform, "detach"):
                alignment_transform = alignment_transform.detach().cpu().numpy()
            pcd.transform(np.asarray(alignment_transform, dtype=np.float64))
        if outlier_enabled and len(pcd.points) > outlier_nb_neighbors:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=outlier_nb_neighbors,
                std_ratio=outlier_std_ratio,
            )
        pcd = pcd.random_down_sample(1.0 / downsample_factor)

        new_xyz = np.asarray(pcd.points)
        new_rgb = np.asarray(pcd.colors)
        if new_xyz.shape[0] == 0:
            raise ValueError("DUSt3R point cloud is empty after downsampling")

        pcd_obj = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd_obj

        fused_point_cloud = torch.from_numpy(new_xyz).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(new_rgb).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001) * point_size
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def create_pcd_from_dust3r_depth(
        self,
        cam,
        depthmaps,
        scale=1.0,
        mask=None,
        init=False,
        pointmap_index=0,
    ):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        dust3r_config = self.config["Training"].get("dust3r", {})
        dust3r_init_config = dust3r_config.get("init", {})
        if init:
            downsample_factor = int(
                dust3r_init_config.get("pcd_downsample", downsample_factor)
            )
            point_size *= float(dust3r_init_config.get("point_size_scale", 1.0))
        depth_min = float(dust3r_config.get("depth_min", 0.05))
        depth_max = float(dust3r_config.get("depth_max", 20.0))
        use_confidence_mask = bool(
            dust3r_init_config.get("use_confidence_mask", True)
        )
        fill_invalid_depth = bool(
            dust3r_init_config.get("fill_invalid_depth", False)
        )
        invalid_depth = float(dust3r_init_config.get("invalid_depth", 2.0))
        invalid_depth_noise = float(
            dust3r_init_config.get("invalid_depth_noise", 0.05)
        )
        sample_stride = int(dust3r_init_config.get("sample_stride", 0))
        gradient_extra_samples = bool(
            dust3r_init_config.get("gradient_extra_samples", False)
        )
        gradient_threshold = float(
            dust3r_init_config.get("gradient_threshold", 0.08)
        )
        gradient_stride = int(dust3r_init_config.get("gradient_stride", 1))
        max_points = int(dust3r_init_config.get("max_points", 200000))
        use_pixel_footprint_scale = bool(
            dust3r_init_config.get("use_pixel_footprint_scale", True)
        )
        pixel_footprint_scale = float(
            dust3r_init_config.get("pixel_footprint_scale", 0.75)
        )
        depth_scale_config = dust3r_init_config.get("depth_scale", {})
        normalize_init_depth_scale = bool(
            depth_scale_config.get("enabled", False)
        )
        depth_scale_mode = depth_scale_config.get("mode", "median")
        target_median_depth = float(depth_scale_config.get("target_median", 2.0))
        target_percentile = float(depth_scale_config.get("target_percentile", 80.0))
        target_percentile_depth = float(
            depth_scale_config.get("target_percentile_depth", target_median_depth)
        )
        min_depth_scale = float(depth_scale_config.get("min_scale", 0.25))
        max_depth_scale = float(depth_scale_config.get("max_scale", 4.0))

        scale = float(scale)
        if not np.isfinite(scale) or abs(scale) < 1e-8:
            scale = 1.0

        depth = depthmaps[pointmap_index]
        depth = depth.detach().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
        depth_t = torch.from_numpy(depth.astype(np.float32))[None, None]
        depth_t = F.interpolate(
            depth_t,
            size=(cam.image_height, cam.image_width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        depth_t = depth_t / scale

        dust3r_valid = torch.isfinite(depth_t)
        dust3r_valid = torch.logical_and(dust3r_valid, depth_t > depth_min)
        dust3r_valid = torch.logical_and(dust3r_valid, depth_t < depth_max)
        if init and normalize_init_depth_scale and dust3r_valid.count_nonzero() > 0:
            valid_depths = depth_t[dust3r_valid]
            if depth_scale_mode == "percentile":
                source_depth = torch.quantile(
                    valid_depths,
                    torch.tensor(
                        np.clip(target_percentile / 100.0, 0.0, 1.0),
                        device=valid_depths.device,
                        dtype=valid_depths.dtype,
                    ),
                )
                target_depth = target_percentile_depth
            else:
                source_depth = torch.median(valid_depths)
                target_depth = target_median_depth
            source_depth_value = float(source_depth.detach().item())
            if np.isfinite(source_depth_value) and source_depth_value > 1e-8:
                depth_scale = source_depth_value / max(target_depth, 1e-8)
                depth_scale = float(
                    np.clip(depth_scale, min_depth_scale, max_depth_scale)
                )
                depth_t = depth_t / depth_scale
                dust3r_valid = torch.isfinite(depth_t)
                dust3r_valid = torch.logical_and(dust3r_valid, depth_t > depth_min)
                dust3r_valid = torch.logical_and(dust3r_valid, depth_t < depth_max)
                Log(
                    "DUSt3R init depth scale normalization: "
                    f"mode={depth_scale_mode}, source={source_depth_value:.4f}, "
                    f"target={target_depth:.4f}, divisor={depth_scale:.4f}"
                )
        if (
            use_confidence_mask
            and mask is not None
            and mask[pointmap_index] is not None
        ):
            mask_np = np.asarray(mask[pointmap_index]).astype(np.float32)
            mask_t = torch.from_numpy(mask_np)[None, None]
            mask_t = F.interpolate(
                mask_t,
                size=(cam.image_height, cam.image_width),
                mode="nearest",
            )[0, 0].bool()
            dust3r_valid = torch.logical_and(dust3r_valid, mask_t)

        valid = dust3r_valid
        if fill_invalid_depth:
            fill_mask = ~dust3r_valid
            if invalid_depth_noise > 0:
                noise = (
                    torch.rand_like(depth_t) - 0.5
                ) * invalid_depth_noise
            else:
                noise = torch.zeros_like(depth_t)
            depth_t = torch.where(fill_mask, invalid_depth + noise, depth_t)
            valid = torch.isfinite(depth_t)
            valid = torch.logical_and(valid, depth_t > 0)
            valid = torch.logical_and(valid, depth_t < depth_max)

        if sample_stride > 1:
            grid_mask = torch.zeros_like(valid, dtype=torch.bool)
            grid_mask[::sample_stride, ::sample_stride] = True
            sample_mask = torch.logical_and(valid, grid_mask)
            if gradient_extra_samples:
                gray = cam.original_image.detach().cpu().mean(dim=0)
                grad_y = torch.zeros_like(gray)
                grad_x = torch.zeros_like(gray)
                grad_y[1:-1, :] = torch.abs(gray[2:, :] - gray[:-2, :]) * 0.5
                grad_x[:, 1:-1] = torch.abs(gray[:, 2:] - gray[:, :-2]) * 0.5
                grad = torch.sqrt(grad_x * grad_x + grad_y * grad_y)
                edge_mask = grad > gradient_threshold
                if gradient_stride > 1:
                    edge_grid = torch.zeros_like(edge_mask, dtype=torch.bool)
                    edge_grid[::gradient_stride, ::gradient_stride] = True
                    edge_mask = torch.logical_and(edge_mask, edge_grid)
                sample_mask = torch.logical_or(
                    sample_mask, torch.logical_and(valid, edge_mask)
                )
            valid = sample_mask

        ys, xs = torch.where(valid)
        if ys.numel() == 0:
            raise ValueError("DUSt3R depth map did not produce valid backprojected points")

        z = depth_t[ys, xs]
        x = (xs.float() - cam.cx) * z / cam.fx
        y = (ys.float() - cam.cy) * z / cam.fy
        points_cam = torch.stack((x, y, z), dim=1).to(device="cuda")

        c2w = torch.linalg.inv(getWorld2View2(cam.R, cam.T)).to(device="cuda")
        points_h = torch.cat(
            (points_cam, torch.ones((points_cam.shape[0], 1), device="cuda")), dim=1
        )
        fused_point_cloud = (points_h @ c2w.transpose(0, 1))[:, :3]

        colors_img = cam.original_image.permute(1, 2, 0).detach().cpu()
        colors = colors_img[ys, xs].float().to(device="cuda")

        if max_points > 0 and fused_point_cloud.shape[0] > max_points:
            perm = torch.randperm(fused_point_cloud.shape[0], device="cuda")[:max_points]
            fused_point_cloud = fused_point_cloud[perm]
            colors = colors[perm]
            z = z.to(device="cuda")[perm]
        elif sample_stride <= 1 and downsample_factor > 1 and fused_point_cloud.shape[0] > downsample_factor:
            keep_count = max(1, fused_point_cloud.shape[0] // downsample_factor)
            perm = torch.randperm(fused_point_cloud.shape[0], device="cuda")[:keep_count]
            fused_point_cloud = fused_point_cloud[perm]
            colors = colors[perm]
            z = z.to(device="cuda")[perm]
        else:
            z = z.to(device="cuda")

        pcd_obj = BasicPointCloud(
            points=fused_point_cloud.detach().cpu().numpy(),
            colors=colors.detach().cpu().numpy(),
            normals=np.zeros((fused_point_cloud.shape[0], 3)),
        )
        self.ply_input = pcd_obj

        fused_color = RGB2SH(colors)
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        if use_pixel_footprint_scale:
            effective_stride = max(1, sample_stride)
            pixel_radius = (
                z
                * max(1.0 / float(cam.fx), 1.0 / float(cam.fy))
                * effective_stride
                * pixel_footprint_scale
            )
            dist2 = torch.clamp_min(pixel_radius * pixel_radius, 0.0000001)
        else:
            dist2 = torch.clamp_min(distCUDA2(fused_point_cloud), 0.0000001) * point_size
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        return fused_point_cloud, features, scales, rots, opacities

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities, kf_id
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None
    ):
        fused_point_cloud, features, scales, rots, opacities = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities, kf_id
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        param_groups = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.lr_delay_mult = training_args.position_lr_delay_mult
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # lr = self.xyz_scheduler_args(iteration)
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    lr_delay_mult=self.lr_delay_mult,
                    max_steps=self.max_steps,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        attrs = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            attrs.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            attrs.append("f_rest_{}".format(i))
        attrs.append("opacity")
        for i in range(self._scaling.shape[1]):
            attrs.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            attrs.append("rot_{}".format(i))
        return attrs

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        def fetchPly_nocolor(path):
            plydata = PlyData.read(path)
            vertices = plydata["vertex"]
            positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
            normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
            colors = np.ones_like(positions)
            return BasicPointCloud(points=positions, colors=colors, normals=normals)

        self.ply_input = fetchPly_nocolor(path)
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        self.unique_kfIDs = torch.zeros((self._xyz.shape[0]))
        self.n_obs = torch.zeros((self._xyz.shape[0]), device="cpu").int()
        self.init_lifecycle(self._xyz.shape[0])

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]
        if self.lifecycle_state.shape[0] == valid_points_mask.shape[0]:
            self.lifecycle_age = self.lifecycle_age[valid_points_mask]
            self.lifecycle_visibility = self.lifecycle_visibility[valid_points_mask]
            self.lifecycle_recent_visibility = self.lifecycle_recent_visibility[
                valid_points_mask
            ]
            self.lifecycle_bad_count = self.lifecycle_bad_count[valid_points_mask]
            self.lifecycle_state = self.lifecycle_state[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()
        self.append_lifecycle(new_xyz.shape[0])

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        lifecycle_config = self.config["Training"].get("lifecycle", {})
        suppress_cold_densify = bool(
            lifecycle_config.get("suppress_cold_densify", False)
        )
        if (
            suppress_cold_densify
            and self.lifecycle_enabled()
            and self.lifecycle_state.shape[0] == grads.shape[0]
        ):
            cold = self.lifecycle_state == 2
            grads[cold] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        protect_newborn_from_prune = bool(
            lifecycle_config.get("protect_newborn_from_prune", False)
        )
        if (
            protect_newborn_from_prune
            and self.lifecycle_enabled()
            and self.lifecycle_state.shape[0] == prune_mask.shape[0]
        ):
            prune_mask = torch.logical_and(prune_mask, self.lifecycle_state != 0)
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
