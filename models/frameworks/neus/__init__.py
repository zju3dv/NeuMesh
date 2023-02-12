import copy

from .neus import NeuS
from models.trainer import Trainer
from models.renderer import SingleRenderer


def get_model(args):

    loss_weights = {
        "img": args.training.loss_weights.setdefault("img", 0.0),
        "mask": args.training.loss_weights.setdefault("mask", 0.0),
        "eikonal": args.training.loss_weights.setdefault("eikonal", 0.0),
        "distill_density": args.training.loss_weights.setdefault(
            "distill_density", 0.0
        ),
        "distill_color": args.training.loss_weights.setdefault("distill_color", 0.0),
        "indicator_reg": args.training.loss_weights.setdefault("indicator_reg", 0.0),
    }

    if loss_weights["mask"] == 0:
        assert (
            "N_outside" in args.model.keys() and args.model.N_outside > 0
        ), "Please specify a positive model:N_outside for neus with nerf++"

    model_config = {
        "obj_bounding_radius": args.model.obj_bounding_radius,
        "W_geo_feat": args.model.setdefault("W_geometry_feature", 256),
        "use_outside_nerf": loss_weights["mask"] == 0,
        "speed_factor": args.training.setdefault("speed_factor", 1.0),
        "variance_init": args.model.setdefault("variance_init", 0.05),
    }

    surface_cfg = {
        "use_siren": args.model.surface.setdefault(
            "use_siren", args.model.setdefault("use_siren", False)
        ),
        "embed_multires": args.model.surface.setdefault("embed_multires", 6),
        "radius_init": args.model.surface.setdefault("radius_init", 1.0),
        "geometric_init": args.model.surface.setdefault("geometric_init", True),
        "D": args.model.surface.setdefault("D", 8),
        "W": args.model.surface.setdefault("W", 256),
        "skips": args.model.surface.setdefault("skips", [4]),
    }

    radiance_cfg = {
        "use_siren": args.model.radiance.setdefault(
            "use_siren", args.model.setdefault("use_siren", False)
        ),
        "embed_multires": args.model.radiance.setdefault("embed_multires", -1),
        "embed_multires_view": args.model.radiance.setdefault(
            "embed_multires_view", -1
        ),
        "use_view_dirs": args.model.radiance.setdefault("use_view_dirs", True),
        "D": args.model.radiance.setdefault("D", 4),
        "W": args.model.radiance.setdefault("W", 256),
        "skips": args.model.radiance.setdefault("skips", []),
    }

    model_config["surface_cfg"] = surface_cfg
    model_config["radiance_cfg"] = radiance_cfg

    model = NeuS(**model_config)

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
        "bounded_near_far": args.model.setdefault("bounded_near_far", False),
    }

    if loss_weights["eikonal"] > 0:
        render_kwargs_train["calc_normal"] = True

    render_kwargs_test = copy.deepcopy(render_kwargs_train)
    render_kwargs_test["rayschunk"] = args.data.val_rayschunk
    render_kwargs_test["perturb"] = False

    trainer = Trainer(
        model,
        loss_weights,
        device_ids=args.device_ids,
        batched=render_kwargs_train["batched"],
    )

    return model, trainer, render_kwargs_train, render_kwargs_test, trainer.renderer
