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
        self.dust3r_device = self.device
        self.dust3r_image_size = 512
        self.dust3r_batch_size = 1
        self.dust3r_use_baseline_ratio_scale = True
        self.dust3r_use_pointmap_scale_sync = True
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
        self.dust3r_max_pointmap_scale_ratio = 2.0
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
        self.dust3r_init_max_pointmap_scale_ratio = 2.0
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
        self.dust3r_refresh_max_tracking_loss_ratio = 1.8
        self.dust3r_refresh_max_depth_change_ratio = 1.8
        self.dust3r_refresh_ema_decay = 0.90
        self.dust3r_refresh_max_calls = 0
        self.dust3r_refresh_event_score_threshold = 1.0
        self.dust3r_refresh_event_joint_bonus = 0.25
        self.last_dust3r_refresh_frame = -1
        self.last_dust3r_refresh_kf_count = -1
        self.dust3r_refresh_call_count = 0
        self.dust3r_first_refresh_done = False
        self.tracking_loss_ema = None
        self.last_dust3r_refresh_depth = None

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.kf_min_interval = int(self.config["Training"].get("kf_min_interval", 0))
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]
        self.max_frames = self.config["Results"].get("max_frames", None)
        tracking_config = self.config.get("Tracking", {})
        self.pose_init_mode = tracking_config.get("pose_init", "previous_pose")
        self.dust3r_config = self.config["Training"].get("dust3r", {})
        self.use_dust3r = bool(self.dust3r_config.get("enabled", False))
        self.dust3r_device = self.dust3r_config.get("device", self.device)
        self.dust3r_image_size = int(self.dust3r_config.get("image_size", 512))
        self.dust3r_batch_size = int(self.dust3r_config.get("batch_size", 1))
        dust3r_scale_config = self.dust3r_config.get("scale", {})
        self.dust3r_use_baseline_ratio_scale = bool(
            dust3r_scale_config.get("baseline_ratio", True)
        )
        self.dust3r_use_pointmap_scale_sync = bool(
            dust3r_scale_config.get("pointmap_sync", True)
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
        pointmap_insert = self.dust3r_config.get("pointmap_insert", {})
        self.dust3r_pointmap_insert_enabled = bool(
            pointmap_insert.get("enabled", True)
        )
        dust3r_selection = self.dust3r_config.get("selection", {})
        self.dust3r_selection_enabled = bool(dust3r_selection.get("enabled", False))
        self.dust3r_candidate_pool = int(dust3r_selection.get("candidate_pool", 4))
        self.dust3r_max_candidate_evals = int(
            dust3r_selection.get("max_candidate_evals", 1)
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
        self.dust3r_max_pointmap_scale_ratio = float(
            dust3r_selection.get("max_pointmap_scale_ratio", 2.0)
        )
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
        self.dust3r_init_max_pointmap_scale_ratio = float(
            dust3r_init_selection.get(
                "max_pointmap_scale_ratio",
                self.dust3r_max_pointmap_scale_ratio,
            )
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
        self.dust3r_refresh_max_tracking_loss_ratio = float(
            dust3r_refresh.get("max_tracking_loss_ratio", 1.8)
        )
        self.dust3r_refresh_max_depth_change_ratio = float(
            dust3r_refresh.get("max_depth_change_ratio", 1.8)
        )
        self.dust3r_refresh_ema_decay = float(dust3r_refresh.get("ema_decay", 0.90))
        self.dust3r_refresh_max_calls = int(dust3r_refresh.get("max_calls", 0))
        health_score = dust3r_refresh.get("health_score", {})
        self.dust3r_refresh_event_score_threshold = float(
            health_score.get("threshold", 1.0)
        )
        self.dust3r_refresh_event_joint_bonus = float(
            health_score.get("joint_bonus", 0.25)
        )
        self.dust3r_refresh_max_tracking_loss_ratio = float(
            health_score.get(
                "loss_trigger_ratio", self.dust3r_refresh_max_tracking_loss_ratio
            )
        )
        self.dust3r_refresh_max_depth_change_ratio = float(
            health_score.get(
                "depth_trigger_ratio", self.dust3r_refresh_max_depth_change_ratio
            )
        )

    def dust3r_refresh_event_score(self, health):
        """Single event score from tracking-loss spike and depth-distribution shift.

        The score uses only two refresh signals:
          D = normalized log depth-ratio change
          L = normalized log tracking-loss ratio spike

        ``max(D, L)`` fires for either a depth or tracking emergency, while
        ``joint_bonus * min(D, L)`` lets two moderate degradations accumulate.
        A score of 1.0 means one signal reached its configured trigger ratio.
        """
        eps = 1e-8

        def normalized_log_ratio(value, trigger_ratio):
            value = max(float(value), eps)
            trigger_ratio = max(float(trigger_ratio), 1.0 + eps)
            return max(0.0, np.log(value) / np.log(trigger_ratio))

        depth_event = normalized_log_ratio(
            health["depth_ratio"], self.dust3r_refresh_max_depth_change_ratio
        )
        loss_event = normalized_log_ratio(
            health["loss_ratio"], self.dust3r_refresh_max_tracking_loss_ratio
        )
        joint_bonus = max(0.0, self.dust3r_refresh_event_joint_bonus)
        score = max(depth_event, loss_event) + joint_bonus * min(
            depth_event, loss_event
        )
        contributions = {
            "depth_ratio": depth_event,
            "loss_ratio": loss_event,
        }
        return score, contributions

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

        valid_candidates = self.get_dust3r_reference_candidates(
            cur_frame_idx,
            candidate_indices,
            self.dust3r_min_baseline,
            self.dust3r_max_baseline,
            candidate_pool=1,
            target_baseline=self.dust3r_min_baseline,
        )
        if not valid_candidates:
            Log(
                f"Skipping DUSt3R for kf {cur_frame_idx}: no reference keyframe "
                f"with baseline in [{self.dust3r_min_baseline:.3f}, "
                f"{self.dust3r_max_baseline:.3f}]"
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
        pointmap_scale_divisors = dust3r_payload.get("pointmap_scale_divisors", [])
        if len(pointmap_scale_divisors) >= 2:
            scale_ratio = abs(
                float(pointmap_scale_divisors[1])
                / max(abs(float(pointmap_scale_divisors[0])), 1e-8)
            )
        else:
            scale_ratio = 1.0
        score = mean_conf * np.log1p(max(match_count, 0)) * max(valid_ratio, 1e-6)
        return {
            "valid_ratio": valid_ratio,
            "mean_conf": mean_conf,
            "match_count": match_count,
            "scale_ratio": scale_ratio,
            "score": float(score),
        }

    def dust3r_payload_passes_quality(self, dust3r_payload, init=False):
        if init:
            min_pair_conf = self.dust3r_init_min_pair_conf
            min_valid_ratio = self.dust3r_init_min_valid_ratio
            min_matches = self.dust3r_init_min_matches
            min_score = self.dust3r_init_min_score
            max_scale_ratio = self.dust3r_init_max_pointmap_scale_ratio
        else:
            min_pair_conf = self.dust3r_min_pair_conf
            min_valid_ratio = self.dust3r_min_valid_ratio
            min_matches = self.dust3r_min_matches
            min_score = self.dust3r_min_score
            max_scale_ratio = self.dust3r_max_pointmap_scale_ratio

        stats = self.summarize_dust3r_payload(dust3r_payload)
        dust3r_payload["quality"] = stats
        checks = [
            stats["mean_conf"] >= min_pair_conf,
            stats["valid_ratio"] >= min_valid_ratio,
            stats["match_count"] >= min_matches,
            stats["score"] >= min_score,
            stats["scale_ratio"] <= max_scale_ratio,
        ]
        if all(checks):
            return True

        Log(
            "Rejecting DUSt3R payload: "
            f"conf={stats['mean_conf']:.3f}/{min_pair_conf:.3f}, "
            f"valid={stats['valid_ratio']:.3f}/{min_valid_ratio:.3f}, "
            f"matches={stats['match_count']}/{min_matches}, "
            f"score={stats['score']:.3f}/{min_score:.3f}, "
            f"scale_ratio={stats['scale_ratio']:.3f}/{max_scale_ratio:.3f}"
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
            ref_candidates = self.get_dust3r_reference_candidates(
                cur_frame_idx,
                candidate_indices,
                self.dust3r_min_baseline,
                self.dust3r_max_baseline,
                candidate_pool=self.dust3r_candidate_pool,
                target_baseline=self.dust3r_target_baseline,
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
                f"scale_ratio {stats['scale_ratio']:.3f}, "
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

    def update_tracking_health(self, render_pkg, _curr_visibility):
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

        health = {
            "tracking_loss": tracking_loss,
            "loss_ratio": loss_ratio,
            "median_depth": median_depth,
            "depth_ratio": depth_ratio,
        }
        return health

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

        if (
            self.dust3r_refresh_force_after_bootstrap
            and not self.dust3r_first_refresh_done
            and cur_frame_idx > self.current_window[-1]
        ):
            return True, "initial_multiview"

        if frame_gap < self.dust3r_refresh_min_frame_gap:
            return False, None
        if keyframe_gap < self.dust3r_refresh_min_keyframe_gap:
            return False, None

        score, contributions = self.dust3r_refresh_event_score(health)
        if score >= self.dust3r_refresh_event_score_threshold:
            if contributions["depth_ratio"] > 0 and contributions["loss_ratio"] > 0:
                reason = "joint_depth_tracking_event"
            elif contributions["depth_ratio"] >= contributions["loss_ratio"]:
                reason = "depth_distribution_shift"
            else:
                reason = "tracking_loss_spike"
            Log(
                "DUSt3R refresh event score "
                f"{score:.3f} >= {self.dust3r_refresh_event_score_threshold:.3f} "
                f"(D={contributions['depth_ratio']:.3f}, "
                f"L={contributions['loss_ratio']:.3f}, "
                f"depth_ratio={health['depth_ratio']:.3f}, "
                f"loss_ratio={health['loss_ratio']:.3f})"
            )
            return True, reason
        return False, None

    def prepare_dust3r_refresh_payload(self, cur_frame_idx, reason):
        if reason == "initial_multiview":
            self.dust3r_first_refresh_done = True
        candidate_indices = [idx for idx in self.current_window if idx != cur_frame_idx]
        if not candidate_indices:
            candidate_indices = [idx for idx in self.kf_indices if idx != cur_frame_idx]

        ref_candidates = self.get_dust3r_reference_candidates(
            cur_frame_idx,
            candidate_indices,
            self.dust3r_refresh_min_baseline,
            self.dust3r_refresh_max_baseline,
            candidate_pool=self.dust3r_refresh_candidate_pool,
            target_baseline=self.dust3r_refresh_target_baseline,
        )
        if not ref_candidates and reason == "initial_multiview" and candidate_indices:
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
        Log(
            f"DUSt3R refresh accepted for frame {cur_frame_idx}: "
            f"{reason}, ref {payload['reference_frame_idx']}"
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

    def estimate_dust3r_pointmap_scale_divisors(
        self,
        poses,
        matches_3d0,
        matches_3d1,
        cur_frame_idx,
        ref_frame_idx,
        fallback_scale_divisor,
    ):
        if not self.dust3r_use_pointmap_scale_sync:
            return [fallback_scale_divisor, fallback_scale_divisor]
        if matches_3d0 is None or matches_3d1 is None:
            return [fallback_scale_divisor, fallback_scale_divisor]
        if len(matches_3d0) < 32 or len(matches_3d1) < 32:
            return [fallback_scale_divisor, fallback_scale_divisor]

        pose0 = np.asarray(poses[0], dtype=np.float64)
        pose1 = np.asarray(poses[1], dtype=np.float64)
        pts0 = np.asarray(matches_3d0, dtype=np.float64)
        pts1 = np.asarray(matches_3d1, dtype=np.float64)
        finite = np.isfinite(pts0).all(axis=1) & np.isfinite(pts1).all(axis=1)
        if finite.sum() < 32:
            return [fallback_scale_divisor, fallback_scale_divisor]
        pts0 = pts0[finite]
        pts1 = pts1[finite]

        max_matches = 4096
        if pts0.shape[0] > max_matches:
            rng = np.random.default_rng(0)
            sample = rng.choice(pts0.shape[0], size=max_matches, replace=False)
            pts0 = pts0[sample]
            pts1 = pts1[sample]

        local0 = (pts0 - pose0[:3, 3]) @ pose0[:3, :3]
        local1 = (pts1 - pose1[:3, 3]) @ pose1[:3, :3]
        valid_depth = (local0[:, 2] > 1e-6) & (local1[:, 2] > 1e-6)
        if valid_depth.sum() < 32:
            return [fallback_scale_divisor, fallback_scale_divisor]
        local0 = local0[valid_depth]
        local1 = local1[valid_depth]

        c2w0 = torch.linalg.inv(
            getWorld2View2(self.cameras[cur_frame_idx].R, self.cameras[cur_frame_idx].T)
        ).detach().cpu().numpy()
        c2w1 = torch.linalg.inv(
            getWorld2View2(self.cameras[ref_frame_idx].R, self.cameras[ref_frame_idx].T)
        ).detach().cpu().numpy()
        vec0 = local0 @ c2w0[:3, :3].T
        vec1 = local1 @ c2w1[:3, :3].T
        baseline = c2w1[:3, 3] - c2w0[:3, 3]

        def solve_scales(mask):
            a = np.stack((vec0[mask], -vec1[mask]), axis=2).reshape(-1, 2)
            b = np.repeat(baseline[None, :], int(mask.sum()), axis=0).reshape(-1)
            try:
                scales, *_ = np.linalg.lstsq(a, b, rcond=None)
            except np.linalg.LinAlgError:
                return None
            return scales

        mask = np.ones(vec0.shape[0], dtype=bool)
        metric_scales = solve_scales(mask)
        if metric_scales is None:
            return [fallback_scale_divisor, fallback_scale_divisor]
        for _ in range(2):
            residual = np.linalg.norm(
                metric_scales[0] * vec0 - metric_scales[1] * vec1 - baseline,
                axis=1,
            )
            median = np.median(residual)
            mad = np.median(np.abs(residual - median)) + 1e-8
            mask = residual < median + 3.0 * 1.4826 * mad
            if mask.sum() < 32:
                break
            refined = solve_scales(mask)
            if refined is None:
                break
            metric_scales = refined

        if (
            not np.isfinite(metric_scales).all()
            or metric_scales[0] <= 1e-8
            or metric_scales[1] <= 1e-8
        ):
            return [fallback_scale_divisor, fallback_scale_divisor]

        scale_divisors = 1.0 / metric_scales
        scale_divisors = np.clip(
            scale_divisors, self.dust3r_scale_min, self.dust3r_scale_max
        )
        pointmap_ratio = scale_divisors[1] / max(scale_divisors[0], 1e-8)
        Log(
            "DUSt3R pointmap scale divisors: "
            f"cur={scale_divisors[0]:.6f}, ref={scale_divisors[1]:.6f}, "
            f"ref/cur={pointmap_ratio:.4f}, matches={int(mask.sum())}"
        )
        return [float(scale_divisors[0]), float(scale_divisors[1])]

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
                poses,
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
        pointmap_scale_divisors = self.estimate_dust3r_pointmap_scale_divisors(
            poses,
            _matches_3d0,
            _matches_3d1,
            cur_frame_idx,
            ref_frame_idx,
            scale_divisor,
        )
        scale_divisor = pointmap_scale_divisors[0]
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
            "poses": poses,
            "depthmaps": depthmaps,
            "match_count": match_count,
            "scale": scale_divisor,
            "pointmap_scale_divisors": pointmap_scale_divisors,
            "reference_idx": reference_idx,
            "reference_frame_idx": ref_frame_idx,
            "world_frame_idx": world_frame_idx,
            "world_R": world_viewpoint.R.detach().clone(),
            "world_T": world_viewpoint.T.detach().clone(),
            "pointmap_indices": [0],
            "init": init,
        }

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

        pose_optimizer = torch.optim.Adam(opt_params)
        tracking_loss_value = None
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
                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                    )
                )
            if converged:
                break

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
                                f"{refresh_reason}, loss_ratio={health['loss_ratio']:.3f}, "
                                f"depth_ratio={health['depth_ratio']:.3f}"
                            )
                            dust3r_payload = self.prepare_dust3r_refresh_payload(
                                cur_frame_idx, refresh_reason
                            )
                            if dust3r_payload is not None:
                                self.last_dust3r_refresh_depth = health["median_depth"]
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
