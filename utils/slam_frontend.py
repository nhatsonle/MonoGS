import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.dust3r_utils import get_result
from utils.slam_utils import get_loss_tracking, get_median_depth


class FrontEnd(mp.Process):
    def __init__(self, config, dust3r_model=None):
        super().__init__()
        self.config = config
        self.dust3r_model = dust3r_model
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False
        self.use_dust3r = False
        self.dust3r_config = {}
        self.dust3r_adaptive = False
        self.dust3r_refresh_only = False
        self.dust3r_device = self.device
        self.dust3r_image_size = 512
        self.dust3r_batch_size = 1
        self.dust3r_use_baseline_ratio_scale = True
        self.dust3r_scale_min = 0.05
        self.dust3r_scale_max = 20.0
        self.dust3r_min_baseline = 0.05
        self.dust3r_max_baseline = 2.0
        self.dust3r_require_initialized = True
        self.dust3r_pointmap_insert_enabled = True
        self.dust3r_selection_enabled = False
        self.dust3r_candidate_pool = 4
        self.dust3r_max_candidate_evals = 1
        self.dust3r_target_baseline = 0.0
        self.dust3r_min_pair_conf = 3.0
        self.dust3r_min_valid_ratio = 0.05
        self.dust3r_min_matches = 128
        self.dust3r_min_score = 0.0
        self.dust3r_min_keyframe_gap = 0
        self.last_dust3r_kf_count = -1
        self.dust3r_calls = 0
        self.dust3r_time = 0.0
        self.dust3r_init_enabled = False
        self.dust3r_init_mode = "pair"
        self.dust3r_init_anchor_idx = None
        self.dust3r_init_max_search_frames = 10
        self.dust3r_init_min_baseline = 0.5
        self.dust3r_init_max_baseline = 20.0
        self.dust3r_init_fallback_to_depth = True
        self.dust3r_init_only = False
        self.dust3r_init_prior_only = False
        self.dust3r_init_backproject = False
        self.dust3r_init_candidate_pool = 4
        self.dust3r_init_max_candidate_evals = 1
        self.dust3r_init_target_baseline = 0.0
        self.dust3r_init_min_pair_conf = 3.0
        self.dust3r_init_min_valid_ratio = 0.05
        self.dust3r_init_min_matches = 128
        self.dust3r_init_min_score = 0.0
        self.dust3r_initialized_from_pair = False
        self.dust3r_refresh_enabled = False
        self.dust3r_refresh_backproject_depth = True
        self.dust3r_refresh_force_after_bootstrap = True
        self.dust3r_refresh_min_frame_gap = 30
        self.dust3r_refresh_min_keyframe_gap = 2
        self.dust3r_refresh_candidate_pool = 4
        self.dust3r_refresh_min_baseline = 0.05
        self.dust3r_refresh_max_baseline = 1.50
        self.dust3r_refresh_target_baseline = 0.25
        self.dust3r_refresh_min_opacity_coverage = 0.18
        self.dust3r_refresh_opacity_threshold = 0.25
        self.dust3r_refresh_max_tracking_loss_ratio = 1.8
        self.dust3r_refresh_max_depth_change_ratio = 1.8
        self.dust3r_refresh_min_visible_gaussian_ratio = 0.01
        self.dust3r_refresh_ema_decay = 0.90
        self.dust3r_refresh_max_calls = 0
        self.dust3r_refresh_loss_threshold = 1.0
        self.dust3r_refresh_photo_weight = 1.0
        self.dust3r_refresh_opacity_weight = 1.0
        self.dust3r_refresh_visibility_weight = 1.0
        self.dust3r_refresh_geometry_weight = 1.0
        self.dust3r_refresh_bootstrap_weight = 1.0
        self.dust3r_adaptive_max_candidate_evals = 1
        self.dust3r_adaptive_target_parallax = 0.12
        self.dust3r_adaptive_min_parallax = 0.03
        self.dust3r_adaptive_max_parallax = 0.40
        self.dust3r_adaptive_loss_warmup = 20
        self.dust3r_adaptive_loss_sigma = 2.5
        self.dust3r_adaptive_min_frame_gap = 20
        self.dust3r_adaptive_min_keyframe_gap = 1
        self.last_dust3r_refresh_frame = -1
        self.last_dust3r_refresh_kf_count = -1
        self.dust3r_refresh_call_count = 0
        self.dust3r_first_refresh_done = False
        self.tracking_loss_ema = None
        self.refresh_loss_history = []
        self.last_dust3r_refresh_depth = None

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        # Camera-tracking pose optimizer: "adam" (1st-order, MonoGS default) or
        # "lbfgs" (quasi-Newton, 2nd-order curvature from the grad_tau history).
        self.tracking_optimizer = self.config["Training"].get(
            "tracking_optimizer", "adam"
        )
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.kf_min_interval = int(self.config["Training"].get("kf_min_interval", 0))
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.max_frames = self.config["Results"].get("max_frames", None)
        tracking_config = self.config.get("Tracking", {})
        self.pose_init_mode = tracking_config.get("pose_init", "previous_pose")
        self.dust3r_config = self.config["Training"].get("dust3r", {})
        self.use_dust3r = bool(self.dust3r_config.get("enabled", False))
        self.dust3r_adaptive = self.dust3r_config.get("mode", "manual") == "adaptive"
        self.dust3r_refresh_only = bool(
            self.dust3r_config.get("refresh_only", False)
        )
        self.dust3r_device = self.dust3r_config.get("device", self.device)
        self.dust3r_image_size = int(self.dust3r_config.get("image_size", 512))
        self.dust3r_batch_size = int(self.dust3r_config.get("batch_size", 1))
        self.dust3r_adaptive_max_candidate_evals = 1
        dust3r_scale_config = self.dust3r_config.get("scale", {})
        self.dust3r_use_baseline_ratio_scale = bool(
            dust3r_scale_config.get("baseline_ratio", True)
        )
        self.dust3r_scale_min = float(self.dust3r_config.get("scale_min", 0.05))
        self.dust3r_scale_max = float(self.dust3r_config.get("scale_max", 20.0))
        self.dust3r_min_baseline = float(
            self.dust3r_config.get("min_baseline", 0.05)
        )
        self.dust3r_max_baseline = float(
            self.dust3r_config.get("max_baseline", 2.0)
        )
        self.dust3r_require_initialized = bool(
            self.dust3r_config.get("require_initialized", True)
        )
        if self.dust3r_adaptive:
            self.dust3r_pointmap_insert_enabled = self.use_dust3r
        else:
            # Legacy ablations can still explicitly disable DUSt3R insertion.
            pointmap_insert = self.dust3r_config.get("pointmap_insert", {})
            self.dust3r_pointmap_insert_enabled = bool(
                pointmap_insert.get("enabled", self.use_dust3r)
            )
        dust3r_selection = self.dust3r_config.get("selection", {})
        self.dust3r_selection_enabled = bool(
            dust3r_selection.get("enabled", self.dust3r_adaptive)
        )
        self.dust3r_candidate_pool = int(dust3r_selection.get("candidate_pool", 4))
        self.dust3r_max_candidate_evals = int(
            dust3r_selection.get(
                "max_candidate_evals",
                self.dust3r_adaptive_max_candidate_evals,
            )
        )
        self.dust3r_target_baseline = float(
            dust3r_selection.get("target_baseline", 0.0)
        )
        self.dust3r_min_pair_conf = float(
            dust3r_selection.get("min_pair_conf", 3.0)
        )
        self.dust3r_min_valid_ratio = float(
            dust3r_selection.get("min_valid_ratio", 0.05)
        )
        self.dust3r_min_matches = int(dust3r_selection.get("min_matches", 128))
        self.dust3r_min_score = float(dust3r_selection.get("min_score", 0.0))
        dust3r_optimization = self.dust3r_config.get("optimization", {})
        self.dust3r_min_keyframe_gap = int(
            dust3r_optimization.get("min_keyframe_gap", 0)
        )
        dust3r_init = self.dust3r_config.get("init", {})
        self.dust3r_init_enabled = bool(dust3r_init.get("enabled", False))
        self.dust3r_init_mode = dust3r_init.get("mode", "pair")
        self.dust3r_init_max_search_frames = int(
            dust3r_init.get("max_search_frames", 10)
        )
        self.dust3r_init_min_baseline = float(
            dust3r_init.get("min_baseline", self.dust3r_min_baseline)
        )
        self.dust3r_init_max_baseline = float(
            dust3r_init.get("max_baseline", self.dust3r_max_baseline)
        )
        self.dust3r_init_fallback_to_depth = bool(
            dust3r_init.get("fallback_to_depth", True)
        )
        self.dust3r_init_only = bool(dust3r_init.get("only", False))
        self.dust3r_init_prior_only = bool(dust3r_init.get("prior_only", False))
        self.dust3r_init_backproject = bool(
            dust3r_init.get("backproject_depth", False)
        )
        dust3r_init_selection = dust3r_init.get("selection", {})
        self.dust3r_init_candidate_pool = int(
            dust3r_init_selection.get("candidate_pool", self.dust3r_candidate_pool)
        )
        self.dust3r_init_max_candidate_evals = int(
            dust3r_init_selection.get(
                "max_candidate_evals", self.dust3r_max_candidate_evals
            )
        )
        self.dust3r_init_target_baseline = float(
            dust3r_init_selection.get(
                "target_baseline", self.dust3r_target_baseline
            )
        )
        self.dust3r_init_min_pair_conf = float(
            dust3r_init_selection.get("min_pair_conf", self.dust3r_min_pair_conf)
        )
        self.dust3r_init_min_valid_ratio = float(
            dust3r_init_selection.get(
                "min_valid_ratio", self.dust3r_min_valid_ratio
            )
        )
        self.dust3r_init_min_matches = int(
            dust3r_init_selection.get("min_matches", self.dust3r_min_matches)
        )
        self.dust3r_init_min_score = float(
            dust3r_init_selection.get("min_score", self.dust3r_min_score)
        )
        dust3r_refresh = self.dust3r_config.get("refresh", {})
        self.dust3r_refresh_enabled = bool(dust3r_refresh.get("enabled", False))
        self.dust3r_refresh_backproject_depth = bool(
            dust3r_refresh.get("backproject_depth", True)
        )
        self.dust3r_refresh_force_after_bootstrap = bool(
            dust3r_refresh.get("force_after_bootstrap", True)
        )
        self.dust3r_refresh_min_frame_gap = int(
            dust3r_refresh.get("min_frame_gap", 30)
        )
        self.dust3r_refresh_min_keyframe_gap = int(
            dust3r_refresh.get("min_keyframe_gap", 2)
        )
        self.dust3r_refresh_candidate_pool = int(
            dust3r_refresh.get("candidate_pool", self.dust3r_candidate_pool)
        )
        self.dust3r_refresh_min_baseline = float(
            dust3r_refresh.get("min_baseline", self.dust3r_min_baseline)
        )
        self.dust3r_refresh_max_baseline = float(
            dust3r_refresh.get("max_baseline", self.dust3r_max_baseline)
        )
        self.dust3r_refresh_target_baseline = float(
            dust3r_refresh.get("target_baseline", self.dust3r_target_baseline)
        )
        self.dust3r_refresh_min_opacity_coverage = float(
            dust3r_refresh.get(
                "min_opacity_coverage",
                0.12 if self.dust3r_adaptive else 0.18,
            )
        )
        self.dust3r_refresh_opacity_threshold = float(
            dust3r_refresh.get("opacity_threshold", 0.25)
        )
        self.dust3r_refresh_max_tracking_loss_ratio = float(
            dust3r_refresh.get(
                "max_tracking_loss_ratio",
                2.2 if self.dust3r_adaptive else 1.8,
            )
        )
        self.dust3r_refresh_max_depth_change_ratio = float(
            dust3r_refresh.get(
                "max_depth_change_ratio",
                2.0 if self.dust3r_adaptive else 1.8,
            )
        )
        self.dust3r_refresh_min_visible_gaussian_ratio = float(
            dust3r_refresh.get("min_visible_gaussian_ratio", 0.01)
        )
        self.dust3r_refresh_ema_decay = float(
            dust3r_refresh.get("ema_decay", 0.95 if self.dust3r_adaptive else 0.90)
        )
        self.dust3r_refresh_max_calls = int(dust3r_refresh.get("max_calls", 0))
        refresh_loss = dust3r_refresh.get("loss", {})
        self.dust3r_refresh_loss_threshold = float(
            refresh_loss.get("threshold", 1.0)
        )
        self.dust3r_refresh_photo_weight = float(
            refresh_loss.get("photometric_weight", 1.0)
        )
        self.dust3r_refresh_opacity_weight = float(
            refresh_loss.get("opacity_weight", 2.0 if self.dust3r_adaptive else 1.0)
        )
        self.dust3r_refresh_visibility_weight = float(
            refresh_loss.get(
                "visibility_weight",
                2.0 if self.dust3r_adaptive else 1.0,
            )
        )
        self.dust3r_refresh_geometry_weight = float(
            refresh_loss.get("geometry_weight", 1.0)
        )
        self.dust3r_refresh_bootstrap_weight = float(
            refresh_loss.get("bootstrap_weight", 1.0)
        )
        adaptive_policy = self.dust3r_config.get("adaptive", {})
        self.dust3r_adaptive_target_parallax = float(
            adaptive_policy.get("target_parallax", 0.12)
        )
        self.dust3r_adaptive_min_parallax = float(
            adaptive_policy.get("min_parallax", 0.03)
        )
        self.dust3r_adaptive_max_parallax = float(
            adaptive_policy.get("max_parallax", 0.40)
        )
        self.dust3r_adaptive_loss_warmup = int(
            adaptive_policy.get("loss_warmup", 20)
        )
        self.dust3r_adaptive_loss_sigma = float(
            adaptive_policy.get("loss_sigma", 2.5)
        )
        self.dust3r_adaptive_min_frame_gap = int(
            adaptive_policy.get("min_frame_gap", 20)
        )
        self.dust3r_adaptive_min_keyframe_gap = int(
            adaptive_policy.get("min_keyframe_gap", 1)
        )

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def clear_frontend_map_state(self, clear_dust3r_anchor=True):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        if clear_dust3r_anchor:
            self.dust3r_init_anchor_idx = None
            self.dust3r_initialized_from_pair = False
            self.last_dust3r_refresh_frame = -1
            self.last_dust3r_refresh_kf_count = -1
            self.dust3r_refresh_call_count = 0
            self.dust3r_first_refresh_done = False
            self.tracking_loss_ema = None
            self.last_dust3r_refresh_depth = None

    def should_use_dust3r_initialization(self):
        return (
            self.monocular
            and self.use_dust3r
            and self.dust3r_model is not None
            and self.dust3r_init_enabled
        )

    def initialize_dust3r_single_view(self, cur_frame_idx, viewpoint):
        self.clear_frontend_map_state(clear_dust3r_anchor=True)
        while not self.backend_queue.empty():
            self.backend_queue.get()

        viewpoint.update_RT(
            torch.eye(3, device=viewpoint.device),
            torch.zeros(3, device=viewpoint.device),
        )
        dust3r_payload = self.prepare_keyframe_dust3r(
            cur_frame_idx, cur_frame_idx, init=True
        )
        if dust3r_payload is None:
            if not self.dust3r_init_fallback_to_depth:
                Log(
                    f"DUSt3R single-view initialization failed at frame "
                    f"{cur_frame_idx}; waiting because fallback_to_depth=False"
                )
                return False
            self.initialize(cur_frame_idx, viewpoint)
            return True

        dust3r_payload = dict(dust3r_payload)
        dust3r_payload["bootstrap"] = "single_view"
        dust3r_payload["backproject_depth"] = self.dust3r_init_backproject
        if not self.dust3r_payload_passes_quality(dust3r_payload, init=True):
            if not self.dust3r_init_fallback_to_depth:
                Log(
                    f"DUSt3R single-view initialization quality failed at frame "
                    f"{cur_frame_idx}; waiting because fallback_to_depth=False"
                )
                return False
            self.initialize(cur_frame_idx, viewpoint)
            return True

        self.kf_indices = [cur_frame_idx]
        self.initialized = True
        self.dust3r_initialized_from_pair = True
        self.dust3r_init_anchor_idx = cur_frame_idx
        self.last_dust3r_refresh_frame = cur_frame_idx
        self.last_dust3r_refresh_kf_count = len(self.kf_indices)
        self.last_dust3r_refresh_depth = None
        self.request_init(cur_frame_idx, viewpoint, None, dust3r_payload)
        self.reset = False
        Log(f"Initialized map request with DUSt3R single-view frame {cur_frame_idx}")
        return True

    def start_dust3r_initialization(self, cur_frame_idx, viewpoint):
        self.clear_frontend_map_state(clear_dust3r_anchor=False)
        while not self.backend_queue.empty():
            self.backend_queue.get()

        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        self.dust3r_init_anchor_idx = cur_frame_idx
        self.dust3r_initialized_from_pair = False
        self.reset = True
        Log(
            f"Staged frame {cur_frame_idx} as DUSt3R initialization anchor"
        )

    def try_dust3r_initialization(self, cur_frame_idx, viewpoint):
        anchor_idx = self.dust3r_init_anchor_idx
        if anchor_idx is None or anchor_idx not in self.cameras:
            self.start_dust3r_initialization(cur_frame_idx, viewpoint)
            return False

        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
        baseline = torch.norm(
            self.get_camera_center(cur_frame_idx) - self.get_camera_center(anchor_idx)
        ).item()
        searched_frames = cur_frame_idx - anchor_idx
        search_exhausted = searched_frames >= self.dust3r_init_max_search_frames

        ref_candidates = self.get_dust3r_initialization_candidates(cur_frame_idx)
        if not ref_candidates and not search_exhausted:
            Log(
                f"Waiting for DUSt3R init baseline: frame {cur_frame_idx}, "
                f"baseline {baseline:.4f}m < {self.dust3r_init_min_baseline:.4f}m"
            )
            return False

        dust3r_payload = None
        if ref_candidates:
            dust3r_payload = self.prepare_best_keyframe_dust3r(
                cur_frame_idx,
                init=True,
                ref_candidates=ref_candidates,
            )

        if dust3r_payload is not None:
            if self.dust3r_init_prior_only:
                dust3r_payload = dict(dust3r_payload)
                dust3r_payload["regularization_only"] = True
                depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
            else:
                self.kf_indices = [cur_frame_idx]
                depth_map = None
                self.initialized = True
            self.dust3r_initialized_from_pair = True
            self.request_init(cur_frame_idx, viewpoint, depth_map, dust3r_payload)
            self.reset = False
            Log(
                f"Initialized map request with DUSt3R "
                f"{'prior' if self.dust3r_init_prior_only else 'pair'} "
                f"{cur_frame_idx}<->{dust3r_payload['reference_frame_idx']} "
                f"(baseline {baseline:.4f}m)"
            )
            return True

        if not self.dust3r_init_fallback_to_depth:
            Log(
                f"DUSt3R initialization failed at frame {cur_frame_idx}; "
                "waiting because fallback_to_depth=False"
            )
            return False

        Log(
            f"DUSt3R initialization unavailable at frame {cur_frame_idx}; "
            "falling back to monocular depth initialization"
        )
        self.initialize(cur_frame_idx, viewpoint)
        return True

    def run_dust3r_pair(self, img1, img2, tag):
        if not self.use_dust3r or self.dust3r_model is None:
            return None

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.time()
        result = get_result(
            img1,
            img2,
            model=self.dust3r_model,
            device=self.dust3r_device,
            batch_size=self.dust3r_batch_size,
            image_size=self.dust3r_image_size,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.dust3r_calls += 1
        self.dust3r_time += time.time() - start_time
        Log(
            f"DUSt3R {tag}: call {self.dust3r_calls}, "
            f"total {self.dust3r_time:.2f}s"
        )
        return result

    def get_camera_center(self, frame_idx):
        viewpoint = self.cameras[frame_idx]
        c2w = torch.linalg.inv(getWorld2View2(viewpoint.R, viewpoint.T))
        return c2w[:3, 3]

    def adaptive_scene_depth(self):
        median_depth = getattr(self, "median_depth", 0.0)
        if hasattr(median_depth, "detach"):
            median_depth = float(median_depth.detach().item())
        else:
            median_depth = float(median_depth)
        if not np.isfinite(median_depth) or median_depth <= 1e-6:
            median_depth = float(
                self.dust3r_config.get("init", {})
                .get("depth_scale", {})
                .get("target_median", 2.0)
            )
        return max(median_depth, 1e-3)

    def adaptive_baseline_window(self):
        scene_depth = self.adaptive_scene_depth()
        min_baseline = self.dust3r_adaptive_min_parallax * scene_depth
        max_baseline = self.dust3r_adaptive_max_parallax * scene_depth
        target_baseline = self.dust3r_adaptive_target_parallax * scene_depth
        min_baseline = float(np.clip(min_baseline, 0.03, 0.50))
        max_baseline = float(np.clip(max_baseline, min_baseline, 2.0))
        target_baseline = float(np.clip(target_baseline, min_baseline, max_baseline))
        return min_baseline, max_baseline, target_baseline

    def get_dust3r_reference_candidates(
        self,
        cur_frame_idx,
        frame_indices,
        min_baseline,
        max_baseline,
        candidate_pool=1,
        target_baseline=0.0,
    ):
        cur_center = self.get_camera_center(cur_frame_idx)
        valid_candidates = []
        for idx in frame_indices:
            if idx == cur_frame_idx or idx not in self.cameras:
                continue
            baseline = torch.norm(cur_center - self.get_camera_center(idx)).item()
            if baseline < min_baseline:
                continue
            if max_baseline > 0 and baseline > max_baseline:
                continue
            valid_candidates.append((idx, baseline))

        if not valid_candidates:
            return []

        if target_baseline <= 0:
            if max_baseline > min_baseline:
                target_baseline = 0.5 * (min_baseline + max_baseline)
            else:
                target_baseline = min_baseline
        valid_candidates.sort(key=lambda item: abs(item[1] - target_baseline))
        if candidate_pool > 0:
            valid_candidates = valid_candidates[:candidate_pool]
        return valid_candidates

    def get_dust3r_initialization_candidates(self, cur_frame_idx):
        candidate_indices = sorted(idx for idx in self.cameras if idx != cur_frame_idx)
        if self.dust3r_adaptive:
            min_baseline, max_baseline, target_baseline = self.adaptive_baseline_window()
            return self.get_dust3r_reference_candidates(
                cur_frame_idx,
                candidate_indices,
                min_baseline,
                max_baseline,
                candidate_pool=self.dust3r_init_candidate_pool,
                target_baseline=target_baseline,
            )
        return self.get_dust3r_reference_candidates(
            cur_frame_idx,
            candidate_indices,
            self.dust3r_init_min_baseline,
            self.dust3r_init_max_baseline,
            candidate_pool=self.dust3r_init_candidate_pool,
            target_baseline=self.dust3r_init_target_baseline,
        )

    def select_dust3r_reference_keyframe(self, cur_frame_idx):
        if not self.dust3r_pointmap_insert_enabled:
            return None
        if self.dust3r_init_only and self.dust3r_initialized_from_pair:
            return None
        if self.dust3r_require_initialized and not self.initialized:
            return None
        if self.dust3r_min_keyframe_gap > 0 and self.last_dust3r_kf_count >= 0:
            keyframe_gap = len(self.kf_indices) - self.last_dust3r_kf_count
            if keyframe_gap < self.dust3r_min_keyframe_gap:
                Log(
                    f"Skipping DUSt3R for kf {cur_frame_idx}: "
                    f"keyframe gap {keyframe_gap} < {self.dust3r_min_keyframe_gap}"
                )
                return None

        candidate_indices = [idx for idx in self.kf_indices if idx != cur_frame_idx]
        if not candidate_indices:
            return None

        if self.dust3r_adaptive:
            min_baseline, max_baseline, target_baseline = self.adaptive_baseline_window()
        else:
            min_baseline = self.dust3r_min_baseline
            max_baseline = self.dust3r_max_baseline
            target_baseline = self.dust3r_min_baseline
        valid_candidates = self.get_dust3r_reference_candidates(
            cur_frame_idx,
            candidate_indices,
            min_baseline,
            max_baseline,
            candidate_pool=1,
            target_baseline=target_baseline,
        )
        if not valid_candidates:
            Log(
                f"Skipping DUSt3R for kf {cur_frame_idx}: no reference keyframe "
                f"with baseline in [{min_baseline:.3f}, {max_baseline:.3f}]"
            )
            return None

        ref_idx, baseline = min(valid_candidates, key=lambda item: item[1])
        Log(
            f"Selected DUSt3R reference kf {ref_idx} for kf {cur_frame_idx} "
            f"(baseline {baseline:.4f}m)"
        )
        return ref_idx

    def summarize_dust3r_payload(self, dust3r_payload):
        masks = dust3r_payload.get("masks", [])
        confs = dust3r_payload.get("confs", [])
        quality_indices = range(len(masks))

        valid_ratios = []
        mean_confs = []
        for idx in quality_indices:
            if idx >= len(masks) or masks[idx] is None:
                continue
            mask = np.asarray(masks[idx]).astype(bool)
            valid_ratios.append(float(np.count_nonzero(mask)) / float(mask.size))
            if idx < len(confs) and confs[idx] is not None:
                conf = np.asarray(confs[idx], dtype=np.float32)
                if mask.any():
                    mean_confs.append(float(conf[mask].mean()))
                else:
                    mean_confs.append(0.0)

        valid_ratio = min(valid_ratios) if valid_ratios else 0.0
        mean_conf = min(mean_confs) if mean_confs else 0.0
        match_count = int(dust3r_payload.get("match_count", 0))
        score = (
            mean_conf
            * np.log1p(max(match_count, 0))
            * max(valid_ratio, 1e-6)
        )
        return {
            "valid_ratio": valid_ratio,
            "mean_conf": mean_conf,
            "match_count": match_count,
            "score": float(score),
        }

    def dust3r_payload_passes_quality(self, dust3r_payload, init=False):
        if self.dust3r_adaptive:
            stats = self.summarize_dust3r_payload(dust3r_payload)
            dust3r_payload["quality"] = stats
            hard_min_matches = 32 if init else 64
            hard_valid_ratio = 0.01 if init else 0.02
            checks = [
                stats["valid_ratio"] >= hard_valid_ratio,
                stats["match_count"] >= hard_min_matches,
                np.isfinite(stats["score"]),
            ]
            if all(checks):
                return True
            Log(
                "Rejecting DUSt3R payload as invalid: "
                f"valid={stats['valid_ratio']:.3f}/{hard_valid_ratio:.3f}, "
                f"matches={stats['match_count']}/{hard_min_matches}, "
                f"score={stats['score']:.3f}"
            )
            return False

        if init:
            min_pair_conf = self.dust3r_init_min_pair_conf
            min_valid_ratio = self.dust3r_init_min_valid_ratio
            min_matches = self.dust3r_init_min_matches
            min_score = self.dust3r_init_min_score
        else:
            min_pair_conf = self.dust3r_min_pair_conf
            min_valid_ratio = self.dust3r_min_valid_ratio
            min_matches = self.dust3r_min_matches
            min_score = self.dust3r_min_score

        stats = self.summarize_dust3r_payload(dust3r_payload)
        dust3r_payload["quality"] = stats
        checks = [
            stats["mean_conf"] >= min_pair_conf,
            stats["valid_ratio"] >= min_valid_ratio,
            stats["match_count"] >= min_matches,
            stats["score"] >= min_score,
        ]
        if all(checks):
            return True

        Log(
            "Rejecting DUSt3R payload: "
            f"conf={stats['mean_conf']:.3f}/{min_pair_conf:.3f}, "
            f"valid={stats['valid_ratio']:.3f}/{min_valid_ratio:.3f}, "
            f"matches={stats['match_count']}/{min_matches}, "
            f"score={stats['score']:.3f}/{min_score:.3f}"
        )
        return False

    def prepare_best_keyframe_dust3r(self, cur_frame_idx, init=False, ref_candidates=None):
        if not self.use_dust3r or self.dust3r_model is None:
            return None
        if not init and not self.dust3r_pointmap_insert_enabled:
            return None
        selection_enabled = self.dust3r_selection_enabled
        max_candidate_evals = (
            self.dust3r_init_max_candidate_evals
            if init
            else self.dust3r_max_candidate_evals
        )
        if ref_candidates is None:
            if self.dust3r_init_only and self.dust3r_initialized_from_pair:
                return None
            if self.dust3r_require_initialized and not self.initialized:
                return None
            candidate_indices = [idx for idx in self.kf_indices if idx != cur_frame_idx]
            if self.dust3r_adaptive:
                min_baseline, max_baseline, target_baseline = (
                    self.adaptive_baseline_window()
                )
            else:
                min_baseline = self.dust3r_min_baseline
                max_baseline = self.dust3r_max_baseline
                target_baseline = self.dust3r_target_baseline
            ref_candidates = self.get_dust3r_reference_candidates(
                cur_frame_idx,
                candidate_indices,
                min_baseline,
                max_baseline,
                candidate_pool=self.dust3r_candidate_pool,
                target_baseline=target_baseline,
            )

        if not ref_candidates:
            return None

        if not selection_enabled:
            ref_frame_idx = ref_candidates[0][0]
            payload = self.prepare_keyframe_dust3r(
                cur_frame_idx, ref_frame_idx, init=init
            )
            if payload is not None:
                self.last_dust3r_kf_count = len(self.kf_indices)
            return payload

        best_payload = None
        best_stats = None
        max_candidate_evals = max(1, max_candidate_evals)
        for ref_frame_idx, baseline in ref_candidates[:max_candidate_evals]:
            payload = self.prepare_keyframe_dust3r(
                cur_frame_idx, ref_frame_idx, init=init
            )
            if payload is None:
                continue
            stats = self.summarize_dust3r_payload(payload)
            payload["quality"] = stats
            Log(
                "DUSt3R candidate quality: "
                f"kf {cur_frame_idx}, ref {ref_frame_idx}, "
                f"baseline {baseline:.4f}m, conf {stats['mean_conf']:.3f}, "
                f"valid {stats['valid_ratio']:.3f}, "
                f"matches {stats['match_count']}, "
                f"score {stats['score']:.3f}"
            )
            if not self.dust3r_payload_passes_quality(payload, init=init):
                continue
            if best_stats is None or stats["score"] > best_stats["score"]:
                best_payload = payload
                best_stats = stats

        if best_payload is not None:
            self.last_dust3r_kf_count = len(self.kf_indices)
            Log(
                "Selected DUSt3R payload: "
                f"kf {cur_frame_idx}, ref {best_payload['reference_frame_idx']}, "
                f"score {best_stats['score']:.3f}"
            )
        return best_payload

    def update_tracking_health(self, render_pkg, curr_visibility):
        tracking_loss = render_pkg.get("tracking_loss", None)
        if tracking_loss is None:
            tracking_loss = 0.0
        tracking_loss = float(tracking_loss)
        prev_ema = self.tracking_loss_ema
        if prev_ema is None:
            loss_ratio = 1.0
            self.tracking_loss_ema = tracking_loss
        else:
            loss_ratio = tracking_loss / max(prev_ema, 1e-8)
            decay = float(np.clip(self.dust3r_refresh_ema_decay, 0.0, 0.999))
            self.tracking_loss_ema = decay * prev_ema + (1.0 - decay) * tracking_loss

        opacity = render_pkg.get("opacity", None)
        if opacity is None:
            opacity_coverage = 1.0
        else:
            opacity_coverage = float(
                (opacity.detach() > self.dust3r_refresh_opacity_threshold)
                .float()
                .mean()
                .item()
            )

        if curr_visibility.numel() == 0:
            visible_ratio = 0.0
        else:
            visible_ratio = float(
                curr_visibility.count_nonzero().item() / curr_visibility.numel()
            )

        median_depth = render_pkg.get("median_depth", self.median_depth)
        if hasattr(median_depth, "detach"):
            median_depth = float(median_depth.detach().item())
        else:
            median_depth = float(median_depth)
        if (
            self.last_dust3r_refresh_depth is None
            or median_depth <= 1e-8
            or self.last_dust3r_refresh_depth <= 1e-8
        ):
            depth_ratio = 1.0
        else:
            depth_ratio = max(
                median_depth / self.last_dust3r_refresh_depth,
                self.last_dust3r_refresh_depth / median_depth,
            )

        photo_norm = max(self.dust3r_refresh_max_tracking_loss_ratio - 1.0, 1e-6)
        photo_loss = max(0.0, loss_ratio - 1.0) / photo_norm
        opacity_loss = max(
            0.0,
            self.dust3r_refresh_min_opacity_coverage - opacity_coverage,
        ) / max(self.dust3r_refresh_min_opacity_coverage, 1e-6)
        visibility_loss = max(
            0.0,
            self.dust3r_refresh_min_visible_gaussian_ratio - visible_ratio,
        ) / max(self.dust3r_refresh_min_visible_gaussian_ratio, 1e-6)
        geometry_loss = max(0.0, np.log(max(depth_ratio, 1.0))) / max(
            np.log(max(self.dust3r_refresh_max_depth_change_ratio, 1.0 + 1e-6)),
            1e-6,
        )
        bootstrap_loss = 1.0 if self.bootstrap_refresh_pending() else 0.0
        map_evidence_loss = (
            self.dust3r_refresh_photo_weight * photo_loss
            + self.dust3r_refresh_opacity_weight * opacity_loss
            + self.dust3r_refresh_visibility_weight * visibility_loss
            + self.dust3r_refresh_geometry_weight * geometry_loss
            + self.dust3r_refresh_bootstrap_weight * bootstrap_loss
        )
        if bootstrap_loss <= 0.0:
            self.refresh_loss_history.append(float(map_evidence_loss))
            max_history = 256
            if len(self.refresh_loss_history) > max_history:
                self.refresh_loss_history = self.refresh_loss_history[-max_history:]

        return {
            "tracking_loss": tracking_loss,
            "loss_ratio": loss_ratio,
            "opacity_coverage": opacity_coverage,
            "visible_ratio": visible_ratio,
            "median_depth": median_depth,
            "depth_ratio": depth_ratio,
            "photo_loss": photo_loss,
            "opacity_loss": opacity_loss,
            "visibility_loss": visibility_loss,
            "geometry_loss": geometry_loss,
            "bootstrap_loss": bootstrap_loss,
            "map_evidence_loss": map_evidence_loss,
        }

    def adaptive_refresh_threshold(self):
        if len(self.refresh_loss_history) < self.dust3r_adaptive_loss_warmup:
            return None
        losses = np.asarray(self.refresh_loss_history, dtype=np.float32)
        median = float(np.median(losses))
        mad = float(np.median(np.abs(losses - median)))
        robust_std = 1.4826 * mad
        threshold = median + self.dust3r_adaptive_loss_sigma * max(robust_std, 1e-4)
        return max(threshold, 0.25)

    def bootstrap_refresh_pending(self):
        return (
            self.dust3r_refresh_force_after_bootstrap
            and not self.dust3r_first_refresh_done
            and len(self.current_window) > 0
        )

    def should_trigger_dust3r_refresh(self, cur_frame_idx, health):
        if (
            not self.dust3r_refresh_enabled
            or not self.use_dust3r
            or self.dust3r_model is None
            or not self.initialized
            or len(self.current_window) == 0
        ):
            return False, None
        if self.dust3r_refresh_max_calls > 0 and (
            self.dust3r_refresh_call_count >= self.dust3r_refresh_max_calls
        ):
            return False, None

        frame_gap = (
            cur_frame_idx - self.last_dust3r_refresh_frame
            if self.last_dust3r_refresh_frame >= 0
            else cur_frame_idx + 1
        )
        keyframe_gap = (
            len(self.kf_indices) - self.last_dust3r_refresh_kf_count
            if self.last_dust3r_refresh_kf_count >= 0
            else len(self.kf_indices)
        )

        if self.dust3r_adaptive:
            threshold = self.adaptive_refresh_threshold()
            if health["bootstrap_loss"] <= 0.0:
                if threshold is None:
                    return False, None
                if health["map_evidence_loss"] < threshold:
                    return False, None
            else:
                threshold = self.dust3r_refresh_loss_threshold
            health["map_evidence_threshold"] = threshold
        else:
            if health["map_evidence_loss"] < self.dust3r_refresh_loss_threshold:
                return False, None
            health["map_evidence_threshold"] = self.dust3r_refresh_loss_threshold

        # The loss decides whether geometry is needed. Frame/keyframe gaps only
        # avoid repeated calls on nearly identical views.
        if health["bootstrap_loss"] <= 0.0:
            min_frame_gap = (
                self.dust3r_adaptive_min_frame_gap
                if self.dust3r_adaptive
                else self.dust3r_refresh_min_frame_gap
            )
            min_keyframe_gap = (
                self.dust3r_adaptive_min_keyframe_gap
                if self.dust3r_adaptive
                else self.dust3r_refresh_min_keyframe_gap
            )
            if frame_gap < min_frame_gap:
                return False, None
            if keyframe_gap < min_keyframe_gap:
                return False, None

        return True, "map_evidence_loss"

    def prepare_dust3r_refresh_payload(self, cur_frame_idx, reason, health=None):
        bootstrap_refresh = (
            health is not None and health.get("bootstrap_loss", 0.0) > 0.0
        )
        if bootstrap_refresh:
            self.dust3r_first_refresh_done = True

        candidate_indices = [idx for idx in self.current_window if idx != cur_frame_idx]
        if not candidate_indices:
            candidate_indices = [idx for idx in self.kf_indices if idx != cur_frame_idx]

        if self.dust3r_adaptive:
            min_baseline, max_baseline, target_baseline = self.adaptive_baseline_window()
        else:
            min_baseline = self.dust3r_refresh_min_baseline
            max_baseline = self.dust3r_refresh_max_baseline
            target_baseline = self.dust3r_refresh_target_baseline
        ref_candidates = self.get_dust3r_reference_candidates(
            cur_frame_idx,
            candidate_indices,
            min_baseline,
            max_baseline,
            candidate_pool=self.dust3r_refresh_candidate_pool,
            target_baseline=target_baseline,
        )
        if not ref_candidates and bootstrap_refresh and candidate_indices:
            ref_idx = candidate_indices[0]
            baseline = torch.norm(
                self.get_camera_center(cur_frame_idx) - self.get_camera_center(ref_idx)
            ).item()
            ref_candidates = [(ref_idx, baseline)]

        if not ref_candidates:
            Log(f"Skipping DUSt3R refresh for frame {cur_frame_idx}: no reference")
            return None

        payload = self.prepare_best_keyframe_dust3r(
            cur_frame_idx,
            init=False,
            ref_candidates=ref_candidates,
        )
        if payload is None:
            Log(f"DUSt3R refresh rejected for frame {cur_frame_idx}: {reason}")
            return None

        payload = dict(payload)
        payload["refresh"] = True
        payload["refresh_reason"] = reason
        payload["backproject_depth"] = self.dust3r_refresh_backproject_depth
        self.last_dust3r_refresh_frame = cur_frame_idx
        self.last_dust3r_refresh_kf_count = len(self.kf_indices)
        self.dust3r_refresh_call_count += 1
        loss_value = health["map_evidence_loss"] if health is not None else 0.0
        Log(
            f"DUSt3R refresh accepted for frame {cur_frame_idx}: "
            f"{reason}={loss_value:.3f}, "
            f"ref {payload['reference_frame_idx']}"
        )
        return payload

    def estimate_dust3r_scale(self, trans_pose, cur_frame_idx, ref_frame_idx):
        if not self.dust3r_use_baseline_ratio_scale:
            return 1.0
        if cur_frame_idx == ref_frame_idx:
            return 1.0

        cur_center = self.get_camera_center(cur_frame_idx)
        ref_center = self.get_camera_center(ref_frame_idx)
        map_dist = torch.norm(cur_center - ref_center).item()
        dust3r_dist = float(np.linalg.norm(trans_pose[:3, 3]))

        if (
            not np.isfinite(map_dist)
            or not np.isfinite(dust3r_dist)
            or map_dist < 1e-6
            or dust3r_dist < 1e-6
        ):
            return 1.0

        scale_divisor = dust3r_dist / map_dist
        clipped_scale_divisor = float(
            np.clip(scale_divisor, self.dust3r_scale_min, self.dust3r_scale_max)
        )
        if abs(clipped_scale_divisor - scale_divisor) > 1e-6:
            Log(
                f"Clipped DUSt3R scale divisor from {scale_divisor:.6f} "
                f"to {clipped_scale_divisor:.6f}"
            )
        return clipped_scale_divisor

    def prepare_keyframe_dust3r(self, cur_frame_idx, ref_frame_idx, init=False):
        if not self.use_dust3r or self.dust3r_model is None:
            return None

        viewpoint = self.cameras[cur_frame_idx]
        ref_viewpoint = self.cameras[ref_frame_idx]
        try:
            (
                trans_pose,
                pts3d,
                imgs,
                masks,
                reference_idx,
                _matches_im0,
                _matches_im1,
                _matches_3d0,
                _matches_3d1,
                _poses,
                depthmaps,
                confs,
                match_count,
            ) = self.run_dust3r_pair(
                viewpoint.original_image,
                ref_viewpoint.original_image,
                tag=f"kf {cur_frame_idx}<->{ref_frame_idx}",
            )
        except Exception as exc:
            Log(f"DUSt3R inference failed for pair {cur_frame_idx}<->{ref_frame_idx}: {exc}")
            return None

        scale_divisor = self.estimate_dust3r_scale(
            trans_pose, cur_frame_idx, ref_frame_idx
        )
        world_frame_idx = cur_frame_idx if reference_idx == 0 else ref_frame_idx
        Log(
            f"DUSt3R keyframe payload: kf {cur_frame_idx}, ref {ref_frame_idx}, "
            f"world {world_frame_idx}, scale divisor {scale_divisor:.4f}"
        )
        world_viewpoint = self.cameras[world_frame_idx]
        return {
            "pts3d": pts3d,
            "imgs": imgs,
            "masks": masks,
            "confs": confs,
            "depthmaps": depthmaps,
            "match_count": match_count,
            "scale": scale_divisor,
            "reference_idx": reference_idx,
            "reference_frame_idx": ref_frame_idx,
            "world_frame_idx": world_frame_idx,
            "world_R": world_viewpoint.R.detach().clone(),
            "world_T": world_viewpoint.T.detach().clone(),
            "pointmap_indices": [0],
            "init": init,
        }

    def _push_tracking_gui(self, viewpoint):
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                current_frame=viewpoint,
                gtcolor=viewpoint.original_image,
                gtdepth=viewpoint.depth
                if not self.monocular
                else np.zeros((viewpoint.image_height, viewpoint.image_width)),
            )
        )

    def _tracking_adam(self, viewpoint, opt_params):
        # Original MonoGS 1st-order tracking: Adam on the SE(3) delta, baking the
        # delta into R/T after every step via update_pose().
        pose_optimizer = torch.optim.Adam(opt_params)
        tracking_loss_value = None
        render_pkg = None
        for tracking_itr in range(self.tracking_itr_num):
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            pose_optimizer.zero_grad()
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            tracking_loss_value = float(loss_tracking.detach().item())
            loss_tracking.backward()

            with torch.no_grad():
                pose_optimizer.step()
                converged = update_pose(viewpoint)

            if tracking_itr % 10 == 0:
                self._push_tracking_gui(viewpoint)
            if converged:
                break
        return render_pkg, tracking_loss_value

    def _tracking_lbfgs(self, viewpoint, opt_params):
        # 2nd-order tracking via L-BFGS: the curvature of the 6-DoF pose is
        # approximated from the history of grad_tau, so each outer step takes a
        # near-Newton stride instead of a small gradient step. A strong-Wolfe
        # line search keeps it stable on the robust (Huber) tracking residual.
        #
        # The SE(3) delta must NOT be baked into R/T while L-BFGS is still
        # probing within one step() (its line search evaluates the closure
        # several times and relies on the delta accumulating). We therefore
        # bake + reset the delta with update_pose() only AFTER each step()
        # completes, then re-linearize for the next outer iteration.
        params = [g["params"][0] for g in opt_params]
        lr = float(self.config["Training"].get("tracking_lbfgs_lr", 1.0))
        inner_iter = int(self.config["Training"].get("tracking_lbfgs_inner_iter", 5))
        # Outer iterations re-bake the delta into R/T (re-linearization point).
        outer_iter = int(
            self.config["Training"].get(
                "tracking_lbfgs_outer_iter",
                max(1, self.tracking_itr_num // inner_iter),
            )
        )

        last = {"loss": None, "render_pkg": None}

        def closure():
            pose_optimizer.zero_grad()
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
            loss_tracking = get_loss_tracking(
                self.config, image, depth, opacity, viewpoint
            )
            loss_tracking.backward()
            last["loss"] = float(loss_tracking.detach().item())
            last["render_pkg"] = render_pkg
            return loss_tracking

        tracking_loss_value = None
        render_pkg = None
        for outer in range(outer_iter):
            pose_optimizer = torch.optim.LBFGS(
                params,
                lr=lr,
                max_iter=inner_iter,
                line_search_fn="strong_wolfe",
            )
            pose_optimizer.step(closure)
            tracking_loss_value = last["loss"]
            render_pkg = last["render_pkg"]

            with torch.no_grad():
                converged = update_pose(viewpoint)

            self._push_tracking_gui(viewpoint)
            if converged:
                break

        # Guarantee a render_pkg even if outer_iter somehow ran zero times.
        if render_pkg is None:
            render_pkg = render(
                viewpoint, self.gaussians, self.pipeline_params, self.background
            )
            tracking_loss_value = float(
                get_loss_tracking(
                    self.config,
                    render_pkg["render"],
                    render_pkg["depth"],
                    render_pkg["opacity"],
                    viewpoint,
                )
                .detach()
                .item()
            )
        return render_pkg, tracking_loss_value

    def tracking(self, cur_frame_idx, viewpoint):
        prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
        if self.pose_init_mode == "constant_velocity" and (
            cur_frame_idx - 2 * self.use_every_n_frames
        ) in self.cameras:
            prev2 = self.cameras[cur_frame_idx - 2 * self.use_every_n_frames]
            w2c_prev = getWorld2View2(prev.R, prev.T)
            w2c_prev2 = getWorld2View2(prev2.R, prev2.T)
            w2c_pred = w2c_prev @ torch.linalg.inv(w2c_prev2) @ w2c_prev
            viewpoint.update_RT(w2c_pred[:3, :3], w2c_pred[:3, 3])
        else:
            viewpoint.update_RT(prev.R, prev.T)

        opt_params = []
        opt_params.append(
            {
                "params": [viewpoint.cam_rot_delta],
                "lr": self.config["Training"]["lr"]["cam_rot_delta"],
                "name": "rot_{}".format(viewpoint.uid),
            }
        )
        opt_params.append(
            {
                "params": [viewpoint.cam_trans_delta],
                "lr": self.config["Training"]["lr"]["cam_trans_delta"],
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

        if self.tracking_optimizer == "lbfgs":
            render_pkg, tracking_loss_value = self._tracking_lbfgs(
                viewpoint, opt_params
            )
        else:
            render_pkg, tracking_loss_value = self._tracking_adam(
                viewpoint, opt_params
            )

        image, depth, opacity = (
            render_pkg["render"],
            render_pkg["depth"],
            render_pkg["opacity"],
        )
        self.median_depth = get_median_depth(depth, opacity)
        render_pkg["tracking_loss"] = tracking_loss_value
        render_pkg["median_depth"] = self.median_depth
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            if kf_idx not in occ_aware_visibility:
                to_remove.append(kf_idx)
                continue
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            if denom == 0:
                to_remove.append(kf_idx)
                continue
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame

    def request_keyframe(
        self, cur_frame_idx, viewpoint, current_window, depthmap, dust3r_payload=None
    ):
        msg = [
            "keyframe",
            cur_frame_idx,
            viewpoint,
            current_window,
            depthmap,
            dust3r_payload,
        ]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map, dust3r_payload=None):
        msg = ["init", cur_frame_idx, viewpoint, depth_map, dust3r_payload]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        for kf_id, kf_R, kf_T in keyframes:
            self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                if cur_frame_idx >= len(self.dataset) or (
                    self.max_frames is not None and cur_frame_idx >= self.max_frames
                ):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    if self.should_use_dust3r_initialization():
                        if self.dust3r_init_mode == "single_view":
                            initialized = self.initialize_dust3r_single_view(
                                cur_frame_idx, viewpoint
                            )
                        else:
                            initialized = self.try_dust3r_initialization(
                                cur_frame_idx, viewpoint
                            )
                        if initialized:
                            self.current_window.append(cur_frame_idx)
                    else:
                        self.initialize(cur_frame_idx, viewpoint)
                        self.current_window.append(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                health = self.update_tracking_health(render_pkg, curr_visibility)
                refresh_requested, refresh_reason = self.should_trigger_dust3r_refresh(
                    cur_frame_idx, health
                )
                create_kf = self.is_keyframe(
                    cur_frame_idx,
                    last_keyframe_idx,
                    curr_visibility,
                    self.occ_aware_visibility,
                )
                if len(self.current_window) < self.window_size:
                    union = torch.logical_or(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    intersection = torch.logical_and(
                        curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                    ).count_nonzero()
                    point_ratio = intersection / union
                    create_kf = (
                        check_time
                        and point_ratio < self.config["Training"]["kf_overlap"]
                    )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if self.kf_min_interval > 0:
                    create_kf = (
                        create_kf
                        and (cur_frame_idx - last_keyframe_idx) >= self.kf_min_interval
                    )
                if refresh_requested:
                    create_kf = True
                if create_kf:
                    self.current_window, removed = self.add_to_window(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.clear_frontend_map_state(clear_dust3r_anchor=True)
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    dust3r_payload = None
                    if self.use_dust3r and self.dust3r_model is not None:
                        if refresh_requested:
                            Log(
                                f"Requesting DUSt3R refresh at frame {cur_frame_idx}: "
                                f"{refresh_reason}={health['map_evidence_loss']:.3f}, "
                                f"threshold={health.get('map_evidence_threshold', 0.0):.3f}, "
                                f"photo={health['photo_loss']:.3f}, "
                                f"opacity={health['opacity_loss']:.3f}, "
                                f"visible={health['visibility_loss']:.3f}, "
                                f"geometry={health['geometry_loss']:.3f}, "
                                f"bootstrap={health['bootstrap_loss']:.3f}"
                            )
                            dust3r_payload = self.prepare_dust3r_refresh_payload(
                                cur_frame_idx, refresh_reason, health=health
                            )
                            if dust3r_payload is not None:
                                self.last_dust3r_refresh_depth = health["median_depth"]
                        elif self.dust3r_refresh_only:
                            pass
                        elif self.dust3r_selection_enabled:
                            dust3r_payload = self.prepare_best_keyframe_dust3r(
                                cur_frame_idx
                            )
                        else:
                            ref_frame_idx = self.select_dust3r_reference_keyframe(
                                cur_frame_idx
                            )
                            if ref_frame_idx is not None:
                                dust3r_payload = self.prepare_keyframe_dust3r(
                                    cur_frame_idx, ref_frame_idx
                                )
                                if dust3r_payload is not None:
                                    self.last_dust3r_kf_count = len(self.kf_indices)
                    self.request_keyframe(
                        cur_frame_idx,
                        viewpoint,
                        self.current_window,
                        depth_map,
                        dust3r_payload,
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break
