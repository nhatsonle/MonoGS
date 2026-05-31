import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getWorld2View2
from gaussian_splatting.utils.loss_utils import l1_loss, ssim
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_mapping


class BackEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gaussians = None
        self.pipeline_params = None
        self.opt_params = None
        self.background = None
        self.cameras_extent = None
        self.frontend_queue = None
        self.backend_queue = None
        self.live_mode = False

        self.pause = False
        self.device = "cuda"
        self.dtype = torch.float32
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.last_sent = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

    def set_hyperparams(self):
        self.save_results = self.config["Results"]["save_results"]

        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = (
            self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        )
        self.mapping_itr_num = self.config["Training"]["mapping_itr_num"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = (
            self.cameras_extent * self.config["Training"]["gaussian_extent"]
        )
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = (
            self.config["Dataset"]["single_thread"]
            if "single_thread" in self.config["Dataset"]
            else False
        )
        dust3r_config = self.config["Training"].get("dust3r", {})
        insertion = dust3r_config.get("insertion", {})
        self.dust3r_insertion_enabled = bool(insertion.get("enabled", True))
        self.dust3r_opacity_threshold = float(
            insertion.get("opacity_threshold", 0.35)
        )
        self.dust3r_rgb_residual_threshold = float(
            insertion.get("rgb_residual_threshold", 0.12)
        )
        self.dust3r_min_opacity_floor = float(insertion.get("min_opacity_floor", 0.08))
        self.dust3r_min_insert_points = int(insertion.get("min_points", 128))
        lifecycle_config = self.config["Training"].get("lifecycle", {})
        self.lifecycle_enabled = bool(lifecycle_config.get("enabled", False))
        self.lifecycle_prune_bad = bool(lifecycle_config.get("prune_bad", True))
        self.lifecycle_log_interval = int(lifecycle_config.get("log_interval", 10))
        dust3r_optimization = dust3r_config.get("optimization", {})
        self.dust3r_optimization_enabled = bool(
            dust3r_optimization.get("enabled", False)
        )
        self.dust3r_mapping_iters_with = int(
            dust3r_optimization.get("mapping_iters_with_dust3r", 5)
        )
        self.dust3r_mapping_iters_without = int(
            dust3r_optimization.get("mapping_iters_without_dust3r", 10)
        )
        self.dust3r_preinit_iters_with = int(
            dust3r_optimization.get("preinit_mapping_iters_with_dust3r", 20)
        )
        self.dust3r_initial_ba_iters_with = int(
            dust3r_optimization.get("initial_ba_iters_with_dust3r", 150)
        )
        self.dust3r_skip_densify_after_insert = int(
            dust3r_optimization.get("skip_densify_after_insert", 0)
        )
        self.dust3r_min_insert_points_for_fast_mapping = int(
            dust3r_optimization.get("min_insert_points_for_fast_mapping", 256)
        )
        self.skip_densify_events = 0

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None):
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map
        )

    def get_c2w_tensor(self, viewpoint):
        w2c = getWorld2View2(viewpoint.R, viewpoint.T)
        return torch.linalg.inv(w2c).to(self.device)

    def add_next_kf_from_dust3r(
        self, frame_idx, viewpoint, dust3r_payload, init=False, depth_map=None
    ):
        if dust3r_payload is None:
            self.add_next_kf(frame_idx, viewpoint, init=init, depth_map=depth_map)
            return False, 0

        if (
            self.dust3r_insertion_enabled
            and not init
            and self.gaussians.get_xyz.shape[0] > 0
        ):
            dust3r_payload = self.apply_dust3r_insertion_mask(
                frame_idx, viewpoint, dust3r_payload
            )
            if dust3r_payload is None:
                self.add_next_kf(frame_idx, viewpoint, init=init, depth_map=depth_map)
                return False, 0

        world_frame_idx = dust3r_payload.get("world_frame_idx", frame_idx)
        world_viewpoint = self.viewpoints.get(world_frame_idx, viewpoint)
        transform = self.get_c2w_tensor(world_viewpoint)

        try:
            fused_point_cloud, features, scales, rots, opacities = (
                self.gaussians.create_pcd_from_dust3r(
                    dust3r_payload["pts3d"],
                    dust3r_payload["imgs"],
                    transform,
                    scale=dust3r_payload.get("scale", 1.0),
                    mask=dust3r_payload.get("masks"),
                    init=init,
                    pointmap_indices=dust3r_payload.get("pointmap_indices", [0]),
                )
            )
            inserted_points = fused_point_cloud.shape[0]
            self.gaussians.extend_from_pcd(
                fused_point_cloud, features, scales, rots, opacities, frame_idx
            )
            use_fast_mapping = (
                inserted_points >= self.dust3r_min_insert_points_for_fast_mapping
            )
            if use_fast_mapping and self.dust3r_skip_densify_after_insert > 0:
                self.skip_densify_events = max(
                    self.skip_densify_events, self.dust3r_skip_densify_after_insert
                )
                Log(
                    f"DUSt3R inserted {inserted_points} Gaussians for kf {frame_idx}; "
                    f"skipping next {self.skip_densify_events} densify events"
                )
            return use_fast_mapping, inserted_points
        except Exception as exc:
            Log(f"DUSt3R Gaussian init failed, falling back to depth init: {exc}")
            self.add_next_kf(frame_idx, viewpoint, init=init, depth_map=depth_map)
            return False, 0

    def apply_dust3r_insertion_mask(self, frame_idx, viewpoint, dust3r_payload):
        render_pkg = render(
            viewpoint, self.gaussians, self.pipeline_params, self.background
        )
        render_rgb = torch.clamp(render_pkg["render"], 0.0, 1.0)
        opacity = render_pkg["opacity"]
        gt_rgb = viewpoint.original_image.to(render_rgb.device)
        residual = torch.mean(torch.abs(render_rgb - gt_rgb), dim=0, keepdim=True)

        pts3d = dust3r_payload["pts3d"]
        pointmap_indices = dust3r_payload.get("pointmap_indices", [0])
        base_masks = dust3r_payload.get("masks")
        if base_masks is None:
            masks = [None] * len(pts3d)
        else:
            masks = [
                None if mask is None else np.asarray(mask).astype(bool).copy()
                for mask in base_masks
            ]

        total_before = 0
        total_after = 0
        for idx in pointmap_indices:
            points = pts3d[idx]
            if hasattr(points, "detach"):
                points = points.detach().cpu().numpy()
            th, tw = points.shape[:2]

            opacity_r = F.interpolate(
                opacity.detach()[None],
                size=(th, tw),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            residual_r = F.interpolate(
                residual.detach()[None],
                size=(th, tw),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            insertion = torch.logical_or(
                opacity_r < self.dust3r_opacity_threshold,
                residual_r > self.dust3r_rgb_residual_threshold,
            )
            insertion = torch.logical_or(
                insertion, opacity_r < self.dust3r_min_opacity_floor
            )
            insertion_np = insertion.detach().cpu().numpy().astype(bool)

            base = masks[idx]
            if base is None:
                base = np.ones((th, tw), dtype=bool)

            total_before += int(base.sum())
            combined = np.logical_and(base, insertion_np)
            total_after += int(combined.sum())
            masks[idx] = combined

        Log(
            f"DUSt3R insertion kf {frame_idx}: kept {total_after}/{total_before} "
            f"points before downsample"
        )
        if total_after < self.dust3r_min_insert_points:
            Log(
                f"Skipping DUSt3R insertion for kf {frame_idx}: "
                f"only {total_after} candidate points"
            )
            return None

        dust3r_payload = dict(dust3r_payload)
        dust3r_payload["masks"] = masks
        return dust3r_payload

    def reset(self):
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.viewpoints = {}
        self.current_window = []
        self.initialized = not self.monocular
        self.keyframe_optimizers = None

        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            (
                image,
                viewspace_point_tensor,
                visibility_filter,
                radii,
                depth,
                opacity,
                n_touched,
            ) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["opacity"],
                render_pkg["n_touched"],
            )
            loss_init = get_loss_mapping(
                self.config, image, depth, viewpoint, opacity, initialization=True
            )
            loss_init.backward()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset or (
                    self.iteration_count == self.opt_params.densify_from_iter
                ):
                    self.gaussians.reset_opacity()

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.occ_aware_visibility[cur_frame_idx] = (n_touched > 0).long()
        Log("Initialized map")
        return render_pkg

    def map(self, current_window, prune=False, iters=1):
        if len(current_window) == 0:
            return

        viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]
        random_viewpoint_stack = []
        frames_to_optimize = self.config["Training"]["pose_window"]

        current_window_set = set(current_window)
        for cam_idx, viewpoint in self.viewpoints.items():
            if cam_idx in current_window_set:
                continue
            random_viewpoint_stack.append(viewpoint)

        for _ in range(iters):
            self.iteration_count += 1
            self.last_sent += 1

            loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []
            n_touched_acm = []

            keyframes_opt = []

            for cam_idx in range(len(current_window)):
                viewpoint = viewpoint_stack[cam_idx]
                keyframes_opt.append(viewpoint)
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )

                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)
                n_touched_acm.append(n_touched)

            for cam_idx in torch.randperm(len(random_viewpoint_stack))[:2]:
                viewpoint = random_viewpoint_stack[cam_idx]
                render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
                (
                    image,
                    viewspace_point_tensor,
                    visibility_filter,
                    radii,
                    depth,
                    opacity,
                    n_touched,
                ) = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    render_pkg["n_touched"],
                )
                loss_mapping += get_loss_mapping(
                    self.config, image, depth, viewpoint, opacity
                )
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            loss_mapping.backward()
            self.gaussians.freeze_cold_gradients()
            gaussian_split = False
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                self.occ_aware_visibility = {}
                for idx in range((len(current_window))):
                    kf_idx = current_window[idx]
                    n_touched = n_touched_acm[idx]
                    self.occ_aware_visibility[kf_idx] = (n_touched > 0).long()

                # # compute the visibility of the gaussians
                # # Only prune on the last iteration and when we have full window
                if prune:
                    if len(current_window) == self.config["Training"]["window_size"]:
                        prune_mode = self.config["Training"]["prune_mode"]
                        prune_coviz = 3
                        self.gaussians.n_obs.fill_(0)
                        for window_idx, visibility in self.occ_aware_visibility.items():
                            self.gaussians.n_obs += visibility.cpu()
                        lifecycle_counts = None
                        if self.lifecycle_enabled:
                            lifecycle_counts = self.gaussians.update_lifecycle(
                                self.gaussians.n_obs
                            )
                        to_prune = None
                        if prune_mode == "odometry":
                            to_prune = self.gaussians.n_obs < 3
                            # make sure we don't split the gaussians, break here.
                        if prune_mode == "slam":
                            # only prune keyframes which are relatively new
                            sorted_window = sorted(current_window, reverse=True)
                            mask = self.gaussians.unique_kfIDs >= sorted_window[2]
                            if not self.initialized:
                                mask = self.gaussians.unique_kfIDs >= 0
                            to_prune = torch.logical_and(
                                self.gaussians.n_obs <= prune_coviz, mask
                            )
                        if to_prune is not None and self.monocular:
                            to_prune = to_prune.cuda()
                            if self.lifecycle_enabled:
                                newborn = self.gaussians.lifecycle_state == 0
                                to_prune = torch.logical_and(to_prune, ~newborn)
                                if self.lifecycle_prune_bad:
                                    bad = self.gaussians.bad_mask()
                                    if bad is not None:
                                        to_prune = torch.logical_or(to_prune, bad)
                            self.gaussians.prune_points(to_prune.cuda())
                            for idx in range((len(current_window))):
                                current_idx = current_window[idx]
                                self.occ_aware_visibility[current_idx] = (
                                    self.occ_aware_visibility[current_idx][~to_prune]
                                )
                        if (
                            lifecycle_counts is not None
                            and self.iteration_count % self.lifecycle_log_interval == 0
                        ):
                            counts = self.gaussians.lifecycle_counts()
                            Log(
                                "Lifecycle: "
                                f"newborn={counts['newborn']} "
                                f"stable={counts['stable']} "
                                f"cold={counts['cold']} "
                                f"bad={counts['bad']} "
                                f"total={self.gaussians.get_xyz.shape[0]}"
                            )
                        if not self.initialized:
                            self.initialized = True
                            Log("Initialized SLAM")
                        # # make sure we don't split the gaussians, break here.
                    return False

                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = (
                    self.iteration_count % self.gaussian_update_every
                    == self.gaussian_update_offset
                )
                if update_gaussian:
                    if self.skip_densify_events > 0:
                        self.skip_densify_events -= 1
                        Log(
                            "Skipping densify after DUSt3R insert; "
                            f"remaining {self.skip_densify_events}"
                        )
                    else:
                        self.gaussians.densify_and_prune(
                            self.opt_params.densify_grad_threshold,
                            self.gaussian_th,
                            self.gaussian_extent,
                            self.size_threshold,
                        )
                        gaussian_split = True

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (
                    not update_gaussian
                ):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)
                    gaussian_split = True

                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(self.iteration_count)
                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                # Pose update
                for cam_idx in range(min(frames_to_optimize, len(current_window))):
                    viewpoint = viewpoint_stack[cam_idx]
                    if viewpoint.uid == 0:
                        continue
                    update_pose(viewpoint)
        return gaussian_split

    def color_refinement(self):
        Log("Starting color refinement")

        iteration_total = 26000
        for iteration in tqdm(range(1, iteration_total + 1)):
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(
                random.randint(0, len(viewpoint_idx_stack) - 1)
            )
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg = render(
                viewpoint_cam, self.gaussians, self.pipeline_params, self.background
            )
            image, visibility_filter, radii = (
                render_pkg["render"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - self.opt_params.lambda_dssim) * (
                Ll1
            ) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss.backward()
            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                self.gaussians.update_learning_rate(iteration)
        Log("Map refinement done")

    def push_to_frontend(self, tag=None):
        self.last_sent = 0
        keyframes = []
        for kf_idx in self.current_window:
            kf = self.viewpoints[kf_idx]
            keyframes.append((kf_idx, kf.R.clone(), kf.T.clone()))
        if tag is None:
            tag = "sync_backend"

        msg = [tag, clone_obj(self.gaussians), self.occ_aware_visibility, keyframes]
        self.frontend_queue.put(msg)

    def run(self):
        while True:
            if self.backend_queue.empty():
                if self.pause:
                    time.sleep(0.01)
                    continue
                if len(self.current_window) == 0:
                    time.sleep(0.01)
                    continue

                if self.single_thread:
                    time.sleep(0.01)
                    continue
                self.map(self.current_window)
                if self.last_sent >= 10:
                    self.map(self.current_window, prune=True, iters=10)
                    self.push_to_frontend()
            else:
                data = self.backend_queue.get()
                if data[0] == "stop":
                    break
                elif data[0] == "pause":
                    self.pause = True
                elif data[0] == "unpause":
                    self.pause = False
                elif data[0] == "color_refinement":
                    self.color_refinement()
                    self.push_to_frontend()
                elif data[0] == "init":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    depth_map = data[3]
                    dust3r_payload = data[4] if len(data) > 4 else None
                    Log("Resetting the system")
                    self.reset()

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.add_next_kf_from_dust3r(
                        cur_frame_idx,
                        viewpoint,
                        dust3r_payload,
                        depth_map=depth_map,
                        init=True,
                    )
                    self.initialize_map(cur_frame_idx, viewpoint)
                    self.push_to_frontend("init")

                elif data[0] == "keyframe":
                    cur_frame_idx = data[1]
                    viewpoint = data[2]
                    current_window = data[3]
                    depth_map = data[4]
                    dust3r_payload = data[5] if len(data) > 5 else None

                    self.viewpoints[cur_frame_idx] = viewpoint
                    self.current_window = current_window
                    used_dust3r, inserted_points = self.add_next_kf_from_dust3r(
                        cur_frame_idx,
                        viewpoint,
                        dust3r_payload,
                        depth_map=depth_map,
                    )

                    opt_params = []
                    frames_to_optimize = self.config["Training"]["pose_window"]
                    iter_per_kf = self.mapping_itr_num if self.single_thread else 10
                    if self.dust3r_optimization_enabled and not self.single_thread:
                        iter_per_kf = (
                            self.dust3r_mapping_iters_with
                            if used_dust3r
                            else self.dust3r_mapping_iters_without
                        )
                    if not self.initialized:
                        if (
                            len(self.current_window)
                            == self.config["Training"]["window_size"]
                        ):
                            frames_to_optimize = (
                                self.config["Training"]["window_size"] - 1
                            )
                            iter_per_kf = 50 if self.live_mode else 300
                            if self.dust3r_optimization_enabled and used_dust3r:
                                iter_per_kf = self.dust3r_initial_ba_iters_with
                            Log("Performing initial BA for initialization")
                        else:
                            iter_per_kf = self.mapping_itr_num
                            if self.dust3r_optimization_enabled and used_dust3r:
                                iter_per_kf = self.dust3r_preinit_iters_with
                    if self.dust3r_optimization_enabled:
                        Log(
                            f"Mapping kf {cur_frame_idx}: {iter_per_kf} iters "
                            f"(dust3r={used_dust3r}, inserted={inserted_points})"
                        )
                    for cam_idx in range(len(self.current_window)):
                        if self.current_window[cam_idx] == 0:
                            continue
                        viewpoint = self.viewpoints[current_window[cam_idx]]
                        if cam_idx < frames_to_optimize:
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_rot_delta],
                                    "lr": self.config["Training"]["lr"]["cam_rot_delta"]
                                    * 0.5,
                                    "name": "rot_{}".format(viewpoint.uid),
                                }
                            )
                            opt_params.append(
                                {
                                    "params": [viewpoint.cam_trans_delta],
                                    "lr": self.config["Training"]["lr"][
                                        "cam_trans_delta"
                                    ]
                                    * 0.5,
                                    "name": "trans_{}".format(viewpoint.uid),
                                }
                            )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_a],
                                "lr": 0.01,
                                "name": "exposure_a_{}".format(viewpoint.uid),
                            }
                        )
                        opt_params.append(
                            {
                                "params": [viewpoint.exposure_b],
                                "lr": 0.01,
                                "name": "exposure_b_{}".format(viewpoint.uid),
                            }
                        )
                    self.keyframe_optimizers = torch.optim.Adam(opt_params)

                    self.map(self.current_window, iters=iter_per_kf)
                    self.map(self.current_window, prune=True)
                    self.push_to_frontend("keyframe")
                else:
                    raise Exception("Unprocessed data", data)
        while not self.backend_queue.empty():
            self.backend_queue.get()
        while not self.frontend_queue.empty():
            self.frontend_queue.get()
        return
