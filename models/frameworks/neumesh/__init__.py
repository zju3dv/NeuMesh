import copy

from models.mesh_grid import *
from .neumesh import NeuMesh
from models.renderer import SingleRenderer
from models.trainer import Trainer
from utils.io_util import load_yaml


def get_model(args):

    model_args = args["model"]

    mesh = o3d.io.read_triangle_mesh(model_args.prior_mesh)
    mesh_grid = MeshGrid(
        mesh, args.device_ids[0], model_args.setdefault("distance_method", "frnn")
    )

    model_config = {
        "speed_factor": args.training.setdefault("speed_factor", 1.0),
        "D_density": model_args.setdefault("D_density", 3),
        "D_color": model_args.setdefault("D_color", 4),
        "W": model_args.setdefault("W", 256),
        "geometry_dim": model_args.get("geometry_dim", 32),
        "color_dim": model_args.setdefault("color_dim", 32),
        "multires_view": model_args.setdefault("multires_view", 4),
        "multires_d": model_args.setdefault("multires_d", 8),
        "multires_fg": model_args.setdefault("multires_fg", 2),
        "multires_ft": model_args.setdefault("multires_ft", 2),
        "enable_nablas_input": model_args.setdefault("enable_nablas_input", False),
    }

    ## render kwargs
    render_kwargs_train = {
        # upsample config
        "N_nograd_samples": args.model.setdefault("N_nograd_samples", 2048),
        "N_upsample_iters": args.model.setdefault("N_upsample_iters", 4),
        "obj_bounding_radius": args.data.setdefault("obj_bounding_radius", 1.0),
        "batched": args.data.batch_size is not None,
        "perturb": args.model.setdefault(
            "perturb", True
        ),  # config whether do stratified sampling
        "white_bkgd": args.model.setdefault("white_bkgd", False),
        "bounded_near_far": model_args.setdefault("bounded_near_far", True),
    }

    model_config["learn_indicator_weight"] = model_args.get(
        "learn_indicator_weight", False
    )

    loss_weights = {
        "img": args.training.loss_weights.setdefault("img", 0.0),
        "mask": args.training.loss_weights.setdefault("mask", 0.0),
        "eikonal": args.training.loss_weights.setdefault("eikonal", 0.0),
        "distill_density": args.training.loss_weights.setdefault(
            "distill_density", 0.0
        ),
        "distill_color": args.training.loss_weights.setdefault("distill_color", 0.0),
        "indicator_reg": args.training.loss_weights.setdefault("indicator_reg", 0.1),
    }

    if loss_weights["eikonal"] > 0:
        render_kwargs_train["calc_normal"] = True

    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test["rayschunk"] = args.data.val_rayschunk
    render_kwargs_test["perturb"] = False

    model = NeuMesh(mesh_grid, **model_config)

    renderer = SingleRenderer(model)

    if (
        "teacher_ckpt" in args.training and args.training.teacher_ckpt is not None
    ) and (
        "teacher_config" in args.training and args.training.teacher_config is not None
    ):
        from models.frameworks import build_framework

        teacher_config = load_yaml(args.training.teacher_config)
        teacher_model, _, _, _, _ = build_framework(
            teacher_config, teacher_config.model.framework
        )
        teacher_state_dict = torch.load(args.training.teacher_ckpt)
        teacher_model.load_state_dict(teacher_state_dict["model"])
        model.ln_s = teacher_model.ln_s
        model.speed_factor = teacher_model.speed_factor
    else:
        teacher_model = None
    trainer = Trainer(
        model,
        loss_weights=loss_weights,
        teacher_model=teacher_model,
        device_ids=args.device_ids,
    )

    return model, trainer, render_kwargs_train, render_kwargs_test, renderer
