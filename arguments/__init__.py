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

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None 
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                elif t == list:
                    group.add_argument("--" + key, default=value, type=type(value[0]), nargs='*')
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = "./model/a1_s1"
        self._images = "images"
        self._resolution = -1
        self.camera_path = ""
        self.image_path = ""
        self.param_path = ""
        self.mesh_path = ""
        self.cloth_mesh_path = ""
        self.uv_path = "./data/a1_s1/a1s1_uv.obj"
        self.white_bkgd = False
        self.smplx_gender = "neutral"
        self.data_device = "cuda"
        self.eval = False
        self.dataset_dir = "./data"
        self.dataset_type = "actorshq"
        self.actor = 1
        self.sequence = 1
        self.subject = 170
        self.train_take = 1
        self.test_take = 5
        self.image_downscale_ratio = 1.0
        self.test_camera_index = [6, 126]
        self.train_frame_start_num = [460, 1]
        self.test_frame_start_num = [460, 1]
        self.trained_model_path = "./output/tracking/a1_s1_460_200"
        self.verts_start_idx = 460
        self.init_params_path = ""
        self.random_init_params = False
        self.init_D = 1.0
        self.init_E = 100.0
        self.min_D = 0.1
        self.max_D = 3.0
        self.min_E = 0.5
        self.max_E = 20.0
        self.min_H = 0.8
        self.max_H = 1.2
        self.split_idx_path = "./data/a1_s1/split_idx.npz"
        self.lbs_w = "optimized_weights"
        self.init_nu = 0.3
        self.init_gamma = 500.0
        self.init_kappa = 500.0
        self.mesh_friction_coeff = 0.5
        self.friction_angle = 40.0
        self.grid_size = 200
        self.substep = 400
        self.output_dir = ""
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00004
        self.position_lr_final = 0.00004
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.verts_lr_init = 0.0
        self.verts_lr_final = 0.0
        self.verts_lr_delay_mult = 0.01
        self.verts_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_lpips = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.random_background = False
        self.threshold_xyz = 1.0
        self.threshold_scale = 0.6
        self.laplacian_type = 1
        self.first_frame_verts_opt = False
        self.lr = 1e-1
        self.lr_D = 1e-2
        self.lr_E = 3e-1
        self.lr_H = 1e-2
        self.log_iters = 1
        self.video_iters = 1
        self.visualize = False
        self.seed = 0
        self.use_wandb = False
        self.wandb_entity = "xxxx"
        self.wandb_project = "MPMAvatar"
        self.wandb_iters = 1
        self.wandb_name = ""
        self.save_name = ""
        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
