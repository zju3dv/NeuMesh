import os
import sys
import time
import functools
from tqdm import tqdm

from models.base import get_optimizer, get_scheduler
from models.frameworks import build_framework

from utils import rend_util, io_util
from utils.dist_util import (
    get_local_rank,
    init_env,
    is_master,
    get_rank,
    get_world_size,
)
from utils.print_fn import log
from utils.logger import Logger
from utils.checkpoints import CheckpointIO
from utils.metric_util import *
from dataio import get_data


import torch
import torch.distributed as dist
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


@torch.no_grad()
def validate(
    it,
    intrinsics,
    c2w,
    target_rgb,
    render_kwargs_test,
    volume_render_fn,
    logger,
    trainer,
):

    # N_rays=-1 for rendering full image
    rays_o, rays_d, select_inds = rend_util.get_rays(
        c2w,
        intrinsics,
        render_kwargs_test["H"],
        render_kwargs_test["W"],
        N_rays=-1,
    )
    rgb, depth_v, ret = volume_render_fn(
        rays_o,
        rays_d,
        # calc_normal=True,
        detailed_output=True,
        **render_kwargs_test,
    )

    to_img = functools.partial(
        rend_util.lin2img,
        H=render_kwargs_test["H"],
        W=render_kwargs_test["W"],
        batched=render_kwargs_test["batched"],
    )

    logger.add_imgs(to_img(target_rgb), "val/gt_rgb", it)
    logger.add_imgs(to_img(rgb), "val/predicted_rgb", it)
    logger.add_imgs(
        to_img((depth_v / (depth_v.max() + 1e-10)).unsqueeze(-1)),
        "val/pred_depth_volume",
        it,
    )
    logger.add_imgs(
        to_img(ret["mask_volume"].unsqueeze(-1)),
        "val/pred_mask_volume",
        it,
    )
    if "depth_surface" in ret:
        logger.add_imgs(
            to_img((ret["depth_surface"] / ret["depth_surface"].max()).unsqueeze(-1)),
            "val/pred_depth_surface",
            it,
        )
    if "mask_surface" in ret:
        logger.add_imgs(
            to_img(ret["mask_surface"].unsqueeze(-1).float()),
            "val/predicted_mask",
            it,
        )
    if hasattr(trainer, "val"):
        trainer.val(logger, ret, to_img, it, render_kwargs_test)

    if "normals_volume" in ret:
        logger.add_imgs(
            to_img(ret["normals_volume"] / 2.0 + 0.5),
            "val/predicted_normals",
            it,
        )


def logging(
    it,
    losses,
    extras,
    logger,
    optimizer,
):

    # -------------------
    # log learning rate
    logger.add("learning rates", "whole", optimizer.param_groups[0]["lr"], it)

    # -------------------
    # log losses
    for k, v in losses.items():
        logger.add("losses", k, v.data.cpu().numpy().item(), it)

    # -------------------
    # log extras
    names = [
        "radiance",
        "alpha",
        "implicit_surface",
        "sigma_out",
        "radiance_out",
        "psnr",
    ]
    if "implicit_nablas_norm" in extras:
        names.append("implicit_nablas_norm")
    for n in names:
        p = "whole"
        # key = "raw.{}".format(n)
        key = n
        if key in extras:
            logger.add(
                "extras_{}".format(n),
                "{}.mean".format(p),
                extras[key].mean().data.cpu().numpy().item(),
                it,
            )
            logger.add(
                "extras_{}".format(n),
                "{}.min".format(p),
                extras[key].min().data.cpu().numpy().item(),
                it,
            )
            logger.add(
                "extras_{}".format(n),
                "{}.max".format(p),
                extras[key].max().data.cpu().numpy().item(),
                it,
            )
            logger.add(
                "extras_{}".format(n),
                "{}.norm".format(p),
                extras[key].norm().data.cpu().numpy().item(),
                it,
            )
    if "scalars" in extras:
        for k, v in extras["scalars"].items():
            logger.add("scalars", k, v.mean(), it)


def train(
    args,
    it,
    indices,
    model_input,
    ground_truth,
    render_kwargs_train,
    trainer,
    optimizer,
    scheduler,
):
    ret = trainer.forward(
        args,
        indices,
        model_input,
        ground_truth,
        render_kwargs_train,
        it,
        train_progress=it / args.training.num_iters,
    )

    losses = ret["losses"]
    extras = ret["extras"]

    for k, v in losses.items():
        losses[k] = torch.mean(v)

    optimizer.zero_grad()
    losses["total"].backward()
    # NOTE: check grad before optimizer.step()
    optimizer.step()
    scheduler.step(it)  # NOTE: important! when world_size is not 1

    return losses, extras


def main_function(args):

    init_env(args)

    # ----------------------------
    # -------- shortcuts ---------
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    i_backup = (
        int(args.training.i_backup // world_size) if args.training.i_backup > 0 else -1
    )
    i_val = int(args.training.i_val // world_size) if args.training.i_val > 0 else -1
    exp_dir = args.training.exp_dir

    device = torch.device("cuda", local_rank)

    # logger
    logger = Logger(
        log_dir=exp_dir,
        img_dir=os.path.join(exp_dir, "imgs"),
        monitoring=args.training.get("monitoring", "tensorboard"),
        monitoring_dir=os.path.join(exp_dir, "events"),
        rank=rank,
        is_master=is_master(),
        multi_process_logging=(world_size > 1),
    )

    log.info("=> Experiments dir: {}".format(exp_dir))

    if is_master():
        # backup codes
        io_util.backup(os.path.join(exp_dir, "backup"))

        # save configs
        io_util.save_config(args, os.path.join(exp_dir, "config.yaml"))

    dataset, val_dataset = get_data(
        args, return_val=True, val_downscale=args.data.get("val_downscale", 4.0)
    )
    print(f"#train: {len(dataset)}, #val: {len(val_dataset)}")

    bs = args.data.get("batch_size", None)
    if args.ddp:
        train_sampler = DistributedSampler(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset, sampler=train_sampler, batch_size=bs
        )
        val_sampler = DistributedSampler(val_dataset)
        valloader = torch.utils.data.DataLoader(
            val_dataset, sampler=val_sampler, batch_size=bs
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=bs,
            shuffle=True,
            pin_memory=args.data.get("pin_memory", False),
        )
        valloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    # create model
    (
        model,
        trainer,
        render_kwargs_train,
        render_kwargs_test,
        volume_render_fn,
    ) = build_framework(args, args.model.framework)
    model.to(device)
    # log.info(model)
    # log.info(args)
    # log.info("=> Nerf params: " + str(train_util.count_trainable_parameters(model)))

    render_kwargs_train["H"] = dataset.H
    render_kwargs_train["W"] = dataset.W
    render_kwargs_test["H"] = val_dataset.H
    render_kwargs_test["W"] = val_dataset.W

    # build optimizer
    optimizer = get_optimizer(args, model)

    # checkpoints
    checkpoint_io = CheckpointIO(
        checkpoint_dir=os.path.join(exp_dir, "ckpts"), allow_mkdir=is_master()
    )
    if world_size > 1:
        dist.barrier()
    # Register modules to checkpoint
    checkpoint_io.register_modules(
        model=model,
        optimizer=optimizer,
    )
    model.ln_s.requires_grad = args.training.setdefault("required_grad_lns", False)

    # Load checkpoints
    load_dict = checkpoint_io.load_file(
        args.training.ckpt_file,
        ignore_keys=args.training.ckpt_ignore_keys,
        only_use_keys=args.training.ckpt_only_use_keys,
        map_location=device,
    )

    print(
        f"[Info] mode.ln_s: {model.ln_s}, model.speed_factor: {model.speed_factor}, required_grad_lns: {model.ln_s.requires_grad}"
    )

    logger.load_stats("stats.p")  # this will be used for plotting
    it = load_dict.get("global_step", 0)
    epoch_idx = load_dict.get("epoch_idx", 0)

    # pretrain if needed. must be after load state_dict, since needs 'is_pretrained' variable to be loaded.
    # ---------------------------------------------
    # -------- init perparation only done in master
    # ---------------------------------------------
    if is_master():
        pretrain_config = {"logger": logger}
        if "lr_pretrain" in args.training:
            pretrain_config["lr"] = args.training.lr_pretrain
            if model.implicit_surface.pretrain_hook(pretrain_config):
                checkpoint_io.save(
                    filename="latest.pt".format(it), global_step=it, epoch_idx=epoch_idx
                )

    # Parallel training
    if args.ddp:
        trainer = DDP(
            trainer,
            device_ids=args.device_ids,
            output_device=local_rank,
            find_unused_parameters=False,
        )

    # build scheduler
    scheduler = get_scheduler(args, optimizer, last_epoch=it - 1)
    t0 = time.time()
    log.info(
        "=> Start training..., it={}, lr={}, in {}".format(
            it, optimizer.param_groups[0]["lr"], exp_dir
        )
    )
    end = it >= args.training.num_iters
    with tqdm(range(args.training.num_iters), disable=not is_master()) as pbar:
        if is_master():
            pbar.update(it)
        while it <= args.training.num_iters and not end:
            try:
                if args.ddp:
                    train_sampler.set_epoch(epoch_idx)
                for (indices, model_input, ground_truth) in dataloader:
                    int_it = int(it // world_size)
                    # -------------------
                    # validate
                    # -------------------
                    if i_val > 0 and int_it % i_val == 0:
                        (val_ind, val_in, val_gt) = next(iter(valloader))
                        intrinsics_val = val_in["intrinsics"].to(device)
                        c2w_val = val_in["c2w"].to(device)
                        target_rgb_val = val_gt["rgb"].to(device)
                        validate(
                            it,
                            intrinsics_val,
                            c2w_val,
                            target_rgb_val,
                            render_kwargs_test,
                            volume_render_fn,
                            logger,
                            trainer,
                        )

                    if it >= args.training.num_iters:
                        end = True
                        break

                    # -------------------
                    # train
                    # -------------------
                    start_time = time.time()
                    losses, extras = train(
                        args,
                        it,
                        indices,
                        model_input,
                        ground_truth,
                        render_kwargs_train,
                        trainer,
                        optimizer,
                        scheduler,
                    )
                    # -------------------
                    # logging
                    # -------------------
                    # done every i_save seconds
                    if (args.training.i_save > 0) and (
                        time.time() - t0 > args.training.i_save
                    ):
                        if is_master():
                            checkpoint_io.save(
                                filename="latest.pt",
                                global_step=it,
                                epoch_idx=epoch_idx,
                            )
                        # this will be used for plotting
                        logger.save_stats("stats.p")
                        t0 = time.time()

                    if is_master():
                        # ----------------------------------------------------------------------------
                        # ------------------- things only done in master -----------------------------
                        # ----------------------------------------------------------------------------
                        pbar.set_postfix(
                            lr=optimizer.param_groups[0]["lr"],
                            loss_total=losses["total"].item(),
                            loss_img=losses["loss_img"].item(),
                        )

                        if i_backup > 0 and int_it % i_backup == 0 and it > 0:
                            checkpoint_io.save(
                                filename="{:08d}.pt".format(it),
                                global_step=it,
                                epoch_idx=epoch_idx,
                            )

                    # ----------------------------------------------------------------------------
                    # ------------------- things done in every child process ---------------------------
                    # ----------------------------------------------------------------------------
                    logging(
                        it,
                        losses,
                        extras,
                        logger,
                        optimizer,
                    )

                    # ---------------------
                    # end of one iteration
                    end_time = time.time()
                    log.debug(
                        "=> One iteration time is {:.2f}".format(end_time - start_time)
                    )

                    it += world_size
                    if is_master():
                        pbar.update(world_size)
                # ---------------------
                # end of one epoch
                epoch_idx += 1

            except KeyboardInterrupt:
                if is_master():
                    checkpoint_io.save(
                        filename="latest.pt".format(it),
                        global_step=it,
                        epoch_idx=epoch_idx,
                    )
                    # this will be used for plotting
                logger.save_stats("stats.p")
                sys.exit()

    if is_master():
        checkpoint_io.save(
            filename="final_{:08d}.pt".format(it), global_step=it, epoch_idx=epoch_idx
        )
        logger.save_stats("stats.p")
        log.info("Everything done.")


if __name__ == "__main__":
    # Arguments
    parser = io_util.create_args_parser()
    parser.add_argument(
        "--ddp", action="store_true", help="whether to use DDP to train."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="master port for multi processing. (if used)",
    )
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    main_function(config)
