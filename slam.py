import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.multiprocessing as mp
import yaml
from munch import munchify

import wandb
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.utils.system_utils import mkdir_p
from gui import gui_utils, slam_gui
from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.dust3r_utils import load_dust3r_model
from utils.eval_utils import eval_ate, eval_rendering, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import FakeQueue
from utils.slam_backend import BackEnd
from utils.slam_frontend import FrontEnd


class SLAM:
    def __init__(self, config, save_dir=None, dust3r_model=None):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        torch.cuda.reset_peak_memory_stats()
        start.record()

        self.config = config
        self.save_dir = save_dir
        model_params = munchify(config["model_params"])
        opt_params = munchify(config["opt_params"])
        pipeline_params = munchify(config["pipeline_params"])
        self.model_params, self.opt_params, self.pipeline_params = (
            model_params,
            opt_params,
            pipeline_params,
        )

        self.live_mode = self.config["Dataset"]["type"] == "realsense"
        self.monocular = self.config["Dataset"]["sensor_type"] == "monocular"
        self.use_spherical_harmonics = self.config["Training"]["spherical_harmonics"]
        self.use_gui = self.config["Results"]["use_gui"]
        if self.live_mode:
            self.use_gui = True
        self.eval_rendering = self.config["Results"]["eval_rendering"]
        self.color_refinement = self.config["Results"].get("color_refinement", True)

        model_params.sh_degree = 3 if self.use_spherical_harmonics else 0

        self.gaussians = GaussianModel(model_params.sh_degree, config=self.config)
        self.gaussians.init_lr(6.0)
        self.dataset = load_dataset(
            model_params, model_params.source_path, config=config
        )

        self.gaussians.training_setup(opt_params)
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        frontend_queue = mp.Queue()
        backend_queue = mp.Queue()

        q_main2vis = mp.Queue() if self.use_gui else FakeQueue()
        q_vis2main = mp.Queue() if self.use_gui else FakeQueue()

        self.config["Results"]["save_dir"] = save_dir
        self.config["Training"]["monocular"] = self.monocular

        self.frontend = FrontEnd(self.config, dust3r_model=dust3r_model)
        self.backend = BackEnd(self.config)

        self.frontend.dataset = self.dataset
        self.frontend.background = self.background
        self.frontend.pipeline_params = self.pipeline_params
        self.frontend.frontend_queue = frontend_queue
        self.frontend.backend_queue = backend_queue
        self.frontend.q_main2vis = q_main2vis
        self.frontend.q_vis2main = q_vis2main
        self.frontend.set_hyperparams()

        self.backend.gaussians = self.gaussians
        self.backend.background = self.background
        self.backend.cameras_extent = 6.0
        self.backend.pipeline_params = self.pipeline_params
        self.backend.opt_params = self.opt_params
        self.backend.frontend_queue = frontend_queue
        self.backend.backend_queue = backend_queue
        self.backend.live_mode = self.live_mode

        self.backend.set_hyperparams()

        self.params_gui = gui_utils.ParamsGUI(
            pipe=self.pipeline_params,
            background=self.background,
            gaussians=self.gaussians,
            q_main2vis=q_main2vis,
            q_vis2main=q_vis2main,
        )

        def tensor_size_mb(tensor):
            if not torch.is_tensor(tensor):
                return 0.0
            return tensor.nelement() * tensor.element_size() / (1024 * 1024)

        def log_final_gaussian_stats(gaussians):
            if gaussians is None:
                Log("Final Gaussian stats unavailable", tag="Eval")
                return

            tensor_names = [
                "_xyz",
                "_features_dc",
                "_features_rest",
                "_scaling",
                "_rotation",
                "_opacity",
                "max_radii2D",
                "xyz_gradient_accum",
                "denom",
                "unique_kfIDs",
                "n_obs",
                "lifecycle_age",
                "lifecycle_visibility",
                "lifecycle_recent_visibility",
                "lifecycle_bad_count",
                "lifecycle_bad_score",
                "lifecycle_state",
            ]
            seen_tensors = set()
            model_mb = 0.0
            for name in tensor_names:
                tensor = getattr(gaussians, name, None)
                if not torch.is_tensor(tensor) or id(tensor) in seen_tensors:
                    continue
                seen_tensors.add(id(tensor))
                model_mb += tensor_size_mb(tensor)

            optimizer_mb = 0.0
            optimizer = getattr(gaussians, "optimizer", None)
            if optimizer is not None:
                seen_tensors = set()
                for state in optimizer.state.values():
                    for value in state.values():
                        if torch.is_tensor(value) and id(value) not in seen_tensors:
                            seen_tensors.add(id(value))
                            optimizer_mb += tensor_size_mb(value)

            Log("Final Gaussian count", gaussians.get_xyz.shape[0], tag="Eval")
            Log("Final Gaussian model memory [MB]", f"{model_mb:.2f}", tag="Eval")
            Log(
                "Final Gaussian optimizer state memory [MB]",
                f"{optimizer_mb:.2f}",
                tag="Eval",
            )

        def log_cuda_memory_stats():
            Log(
                "CUDA memory allocated [MB]",
                f"{torch.cuda.memory_allocated() / (1024 * 1024):.2f}",
                tag="Eval",
            )
            Log(
                "CUDA memory reserved [MB]",
                f"{torch.cuda.memory_reserved() / (1024 * 1024):.2f}",
                tag="Eval",
            )
            Log(
                "CUDA max memory allocated [MB]",
                f"{torch.cuda.max_memory_allocated() / (1024 * 1024):.2f}",
                tag="Eval",
            )
            Log(
                "CUDA max memory reserved [MB]",
                f"{torch.cuda.max_memory_reserved() / (1024 * 1024):.2f}",
                tag="Eval",
            )

        backend_process = mp.Process(target=self.backend.run)
        if self.use_gui:
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(5)

        backend_process.start()
        self.frontend.run()
        backend_queue.put(["pause"])

        if hasattr(self.frontend, "dust3r_calls"):
            Log("DUSt3R calls", self.frontend.dust3r_calls, tag="Eval")
            Log("DUSt3R total time", self.frontend.dust3r_time, tag="Eval")

        end.record()
        torch.cuda.synchronize()
        # empty the frontend queue
        N_frames = len(self.frontend.cameras)
        FPS = N_frames / (start.elapsed_time(end) * 0.001)
        Log("Total time", start.elapsed_time(end) * 0.001, tag="Eval")
        Log("Total FPS", N_frames / (start.elapsed_time(end) * 0.001), tag="Eval")

        if self.eval_rendering:
            self.gaussians = self.frontend.gaussians
            kf_indices = self.frontend.kf_indices
            ATE = eval_ate(
                self.frontend.cameras,
                self.frontend.kf_indices,
                self.save_dir,
                0,
                final=True,
                monocular=self.monocular,
            )

            rendering_result = eval_rendering(
                self.frontend.cameras,
                self.gaussians,
                self.dataset,
                self.save_dir,
                self.pipeline_params,
                self.background,
                kf_indices=kf_indices,
                iteration="before_opt",
            )
            columns = ["tag", "psnr", "ssim", "lpips", "RMSE ATE", "FPS"]
            metrics_table = wandb.Table(columns=columns)
            metrics_table.add_data(
                "Before",
                rendering_result["mean_psnr"],
                rendering_result["mean_ssim"],
                rendering_result["mean_lpips"],
                ATE,
                FPS,
            )

            if self.color_refinement:
                # re-used the frontend queue to retrive the gaussians from the backend.
                while not frontend_queue.empty():
                    frontend_queue.get()
                backend_queue.put(["color_refinement"])
                while True:
                    if frontend_queue.empty():
                        time.sleep(0.01)
                        continue
                    data = frontend_queue.get()
                    if data[0] == "sync_backend" and frontend_queue.empty():
                        gaussians = data[1]
                        self.gaussians = gaussians
                        break

                rendering_result = eval_rendering(
                    self.frontend.cameras,
                    self.gaussians,
                    self.dataset,
                    self.save_dir,
                    self.pipeline_params,
                    self.background,
                    kf_indices=kf_indices,
                    iteration="after_opt",
                )
                metrics_table.add_data(
                    "After",
                    rendering_result["mean_psnr"],
                    rendering_result["mean_ssim"],
                    rendering_result["mean_lpips"],
                    ATE,
                    FPS,
                )
            wandb.log({"Metrics": metrics_table})
            save_gaussians(self.gaussians, self.save_dir, "final_after_opt", final=True)

        final_gaussians = self.gaussians if self.eval_rendering else self.frontend.gaussians
        log_final_gaussian_stats(final_gaussians)
        log_cuda_memory_stats()

        backend_queue.put(["stop"])
        backend_process.join()
        Log("Backend stopped and joined the main thread")
        if self.use_gui:
            q_main2vis.put(gui_utils.GaussianPacket(finish=True))
            gui_process.join()
            Log("GUI Stopped and joined the main thread")

    def run(self):
        pass


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config", type=str)
    parser.add_argument("--eval", action="store_true")

    args = parser.parse_args(sys.argv[1:])

    mp.set_start_method("spawn")

    with open(args.config, "r") as yml:
        config = yaml.safe_load(yml)

    config = load_config(args.config)
    save_dir = None

    if args.eval:
        Log("Running MonoGS in Evaluation Mode")
        Log("Following config will be overriden")
        Log("\tsave_results=True")
        config["Results"]["save_results"] = True
        Log("\tuse_gui=False")
        config["Results"]["use_gui"] = False
        Log("\teval_rendering=True")
        config["Results"]["eval_rendering"] = True
        Log("\tuse_wandb=True")
        config["Results"]["use_wandb"] = True

    if config["Results"]["save_results"]:
        mkdir_p(config["Results"]["save_dir"])
        current_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        path = config["Dataset"]["dataset_path"].split("/")
        save_dir = os.path.join(
            config["Results"]["save_dir"], path[-3] + "_" + path[-2], current_datetime
        )
        tmp = args.config
        tmp = tmp.split(".")[0]
        config["Results"]["save_dir"] = save_dir
        mkdir_p(save_dir)
        with open(os.path.join(save_dir, "config.yml"), "w") as file:
            documents = yaml.dump(config, file)
        Log("saving results in " + save_dir)
        run = wandb.init(
            project="MonoGS",
            name=f"{tmp}_{current_datetime}",
            config=config,
            mode=None if config["Results"]["use_wandb"] else "disabled",
        )
        wandb.define_metric("frame_idx")
        wandb.define_metric("ate*", step_metric="frame_idx")

    dust3r_model = None
    dust3r_config = config["Training"].get("dust3r", {})
    if dust3r_config.get("enabled", False):
        checkpoint_path = os.path.expanduser(
            dust3r_config.get(
                "checkpoint", "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
            )
        )
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.abspath(checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                "DUSt3R checkpoint not found: "
                f"{checkpoint_path}. Update Training.dust3r.checkpoint in config."
            )
        device = dust3r_config.get("device", "cuda")
        Log(f"Loading DUSt3R model from {checkpoint_path}")
        dust3r_model = load_dust3r_model(checkpoint_path, device=device)

    slam = SLAM(config, save_dir=save_dir, dust3r_model=dust3r_model)

    slam.run()
    wandb.finish()

    # All done
    Log("Done.")
