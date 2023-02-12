import os

from utils import io_util, rend_util
from utils.checkpoints import sorted_ckpts
from utils.print_fn import log

from dataio import get_data

from models.frameworks import build_framework

import os
import imageio
import numpy as np
from tqdm import tqdm
import open3d as o3d
import cv2
import torch


def normalize(vec, axis=-1):
    return vec / (np.linalg.norm(vec, axis=axis, keepdims=True) + 1e-9)


def view_matrix(forward: np.ndarray, up: np.ndarray, cam_location: np.ndarray):
    rot_z = normalize(forward)
    rot_x = normalize(np.cross(up, rot_z))
    rot_y = normalize(np.cross(rot_z, rot_x))
    mat = np.stack((rot_x, rot_y, rot_z, cam_location), axis=-1)
    hom_vec = np.array([[0.0, 0.0, 0.0, 1.0]])
    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])
    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    forward = poses[:, :3, 2].sum(0)
    up = poses[:, :3, 1].sum(0)
    c2w = view_matrix(forward, up, center)
    return c2w


def look_at(
    cam_location: np.ndarray,
    point: np.ndarray,
    up=np.array([0.0, -1.0, 0.0])  # openCV convention
    # up=np.array([0., 1., 0.])         # openGL convention
):
    # Cam points in positive z direction
    forward = normalize(point - cam_location)  # openCV convention
    # forward = normalize(cam_location - point)   # openGL convention
    return view_matrix(forward, up, cam_location)


def c2w_track_spiral(
    c2w,
    up_vec,
    rads,
    focus: float,
    zrate: float,
    rots: int,
    N: int,
    zdelta: float = 0.0,
):
    # TODO: support zdelta
    """generate camera to world matrices of spiral track, looking at the same point [0,0,focus]

    Args:
        c2w ([4,4] or [3,4]):   camera to world matrix (of the spiral center, with average rotation and average translation)
        up_vec ([3,]):          vector pointing up
        rads ([3,]):            radius of x,y,z direction, of the spiral track
        # zdelta ([float]):       total delta z that is allowed to change
        focus (float):          a focus value (to be looked at) (in camera coordinates)
        zrate ([float]):        a factor multiplied to z's angle
        rots ([int]):           number of rounds to rotate
        N ([int]):              number of total views
    """

    c2w_tracks = []
    rads = np.array(list(rads) + [1.0])

    # focus_in_cam = np.array([0, 0, -focus, 1.])   # openGL convention
    focus_in_cam = np.array([0, 0, focus, 1.0])  # openCV convention
    focus_in_world = np.dot(c2w[:3, :4], focus_in_cam)

    for theta in np.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        cam_location = np.dot(
            c2w[:3, :4],
            # np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads    # openGL convention
            np.array([np.cos(theta), np.sin(theta), np.sin(theta * zrate), 1.0])
            * rads,  # openCV convention
        )
        c2w_i = look_at(cam_location, focus_in_world, up=up_vec)
        c2w_tracks.append(c2w_i)
    return c2w_tracks


def render_function(args, render_kwargs_test, render_fn, ckpt_file=""):

    io_util.cond_mkdir("./out")

    if args.dataset_split is not None:
        args.data.split = args.dataset_split
    if args.background is not None:
        render_kwargs_test["white_bkgd"] = args.background == 1
    dataset = get_data(args, downscale=args.downscale)

    (_, model_input, ground_truth) = dataset[0]
    intrinsics = model_input["intrinsics"].cuda()
    H, W = (dataset.H, dataset.W)
    # NOTE: fx, fy should be scalec with the same ratio. Different ratio will cause the picture itself be stretched.
    #       fx=intrinsics[0,0]                   fy=intrinsics[1,1]
    #       cy=intrinsics[1,2] for H's scal      cx=intrinsics[0,2] for W's scale
    if args.H is not None:
        intrinsics[1, 2] *= args.H / dataset.H
        H = args.H
    if args.H_scale is not None:
        H = int(dataset.H * args.H_scale)
        intrinsics[1, 2] *= H / dataset.H

    if args.W is not None:
        intrinsics[0, 2] *= args.W / dataset.W
        W = args.W
    if args.W_scale is not None:
        W = int(dataset.W * args.W_scale)
        intrinsics[0, 2] *= W / dataset.W
    log.info("=> Rendering resolution @ [{} x {}]".format(H, W))

    c2ws = torch.stack(dataset.c2w_all, dim=0).data.cpu().numpy()

    # -----------------
    # Spiral path
    #   original nerf-like spiral path
    # -----------------
    if args.camera_path == "spiral":

        if args.test_frame is not None:
            test_pose = c2ws[args.test_frame]
            up = test_pose[:3, 1]
            focus_distance = np.linalg.norm(test_pose[:3, 3], axis=-1)
        else:
            test_pose = poses_avg(c2ws)
            focus_distance = np.mean(np.linalg.norm(c2ws[:, :3, 3], axis=-1))
            up = c2ws[:, :3, 1].sum(0)

        rads = np.array(
            [
                np.percentile(np.abs(c2ws[:, 0, 3]), 10, 0),
                np.percentile(np.abs(c2ws[:, 1, 3]), 15, 0),
                np.percentile(np.abs(c2ws[:, 2, 3]), 30, 0),
            ]
        ).reshape(-1)

        if len(args.spiral_rad) >= 1 and args.spiral_rad[0] >= 0:
            rads[0] = args.spiral_rad[0]
        if len(args.spiral_rad) >= 2 and args.spiral_rad[1] >= 0:
            rads[1] = args.spiral_rad[1]
        if len(args.spiral_rad) >= 3 and args.spiral_rad[2] >= 0:
            rads[2] = args.spiral_rad[2]

        print("rads: ", rads)
        render_c2ws = c2w_track_spiral(
            test_pose,
            normalize(up),
            rads,
            focus_distance * 0.8,
            zrate=0.0,
            rots=1,
            N=args.num_views,
        )
        view_list = np.arange(len(render_c2ws))
    else:
        raise RuntimeError("Please choose render type between [spiral]")
    log.info("=> Camera path: {}".format(args.camera_path))

    rgb_imgs = []
    depth_imgs = []
    normal_imgs = []
    # save mesh render images
    render_kwargs_test["rayschunk"] = args.rayschunk

    def integerify(img):
        return (img * 255.0).astype(np.uint8)

    if args.outbase is None:
        outbase = args.expname
    else:
        outbase = args.outbase
    output_dir = os.path.join("out", outbase)

    if not args.outdirectory is None:
        output_dir = os.path.join(output_dir, args.outdirectory)
    os.makedirs(output_dir, exist_ok=True)

    normal_dir = os.path.join(output_dir, "normal")
    os.makedirs(normal_dir, exist_ok=True)

    assert len(render_c2ws) == len(view_list)
    for idx, c2w in enumerate(tqdm(render_c2ws, desc="rendering...")):
        if not args.disable_rgb:
            rays_o, rays_d, select_inds = rend_util.get_rays(
                torch.from_numpy(c2w).float().cuda()[None, ...],
                intrinsics[None, ...],
                H,
                W,
                N_rays=-1,
            )
            with torch.no_grad():
                # NOTE: detailed_output set to False to save a lot of GPU memory.
                rgb, depth, extras = render_fn(
                    rays_o,
                    rays_d,
                    show_progress=True,
                    # calc_normal=True,
                    detailed_output=False,
                    **render_kwargs_test
                )
                depth = depth.data.cpu().reshape(H, W, 1).numpy()
                depth = depth / depth.max()
                rgb_imgs.append(rgb.data.cpu().reshape(H, W, 3).numpy())
                depth_imgs.append(depth)
                b_save_normal = True
                if "normals_volume" not in extras:
                    b_save_normal = False
                if b_save_normal == True:
                    normals = extras["normals_volume"]
                    normals = normals.data.cpu().reshape(H, W, 3).numpy()
                    # if True:
                    #     # (c2w^(-1) @ n)^T = n^T @ c2w^(-1)^T = n^T @ c2w
                    #     normals = normals @ c2w[:3, :3]
                    normal_imgs.append(normals / 2.0 + 0.5)
                img = integerify(rgb_imgs[-1])
                img[..., [0, 2]] = img[..., [2, 0]]
                cv2.imwrite(
                    os.path.join(
                        output_dir,
                        "{}_rgb_{:03d}.png".format(outbase, view_list[idx]),
                    ),
                    img,
                )
                if b_save_normal == True:
                    imageio.imwrite(
                        os.path.join(
                            normal_dir,
                            "{}_normal_{:03d}.png".format(outbase, view_list[idx]),
                        ),
                        integerify(normal_imgs[-1]),
                    )

    rgb_imgs = [integerify(img) for img in rgb_imgs]
    depth_imgs = [integerify(img) for img in depth_imgs]
    normal_imgs = [integerify(img) for img in normal_imgs]

    post_fix = "{}x{}_{}_{}".format(H, W, args.num_views, args.camera_path)
    if not args.disable_rgb:
        imageio.mimwrite(
            os.path.join(output_dir, "{}_rgb_{}.mp4".format(outbase, post_fix)),
            rgb_imgs,
            fps=args.fps,
            quality=10,
        )
        imageio.mimwrite(
            os.path.join(output_dir, "{}_depth_{}.mp4".format(outbase, post_fix)),
            depth_imgs,
            fps=args.fps,
            quality=10,
        )


def main_function(args):
    (
        model,
        trainer,
        render_kwargs_train,
        render_kwargs_test,
        render_fn,
    ) = build_framework(args, args.model.framework)

    if args.load_pt is None:
        # automatically load 'final_xxx.pt' or 'latest.pt'
        ckpt_file = sorted_ckpts(os.path.join(args.training.exp_dir, "ckpts"))[-1]
    else:
        ckpt_file = args.load_pt

    log.info("=> Use ckpt:" + str(ckpt_file))
    state_dict = torch.load(ckpt_file, map_location=args.device)
    model.load_state_dict(state_dict["model"])
    model.to(args.device)
    render_function(args, render_kwargs_test, render_fn, ckpt_file)


def create_render_args(parser):
    parser.add_argument("--num_views", type=int, default=90)
    parser.add_argument("--device", type=str, default="cuda", help="render device")
    parser.add_argument("--downscale", type=float, default=1)
    parser.add_argument("--rayschunk", type=int, default=4096)
    parser.add_argument(
        "--camera_path",
        type=str,
        default="spiral",
        help="choose between [spiral]",
    )
    parser.add_argument("--load_pt", type=str, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--H_scale", type=float, default=None)
    parser.add_argument("--W", type=int, default=None)
    parser.add_argument("--W_scale", type=float, default=None)
    parser.add_argument("--disable_rgb", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--outbase", type=str, default=None, help="base of output filename"
    )
    parser.add_argument("--outdirectory", type=str, default=None)
    parser.add_argument("--background", type=int, default=None)
    parser.add_argument("--test_frame", type=int, default=None)
    parser.add_argument("--spiral_rad", type=float, nargs="+", default=[])
    parser.add_argument("--dataset_split", default="entire", type=str)
    parser.add_argument(
        "--camera_inds",
        type=str,
        help="params for generating camera paths",
        default="0~-1",
    )
    return parser


if __name__ == "__main__":
    # Arguments
    parser = io_util.create_args_parser()
    parser = create_render_args(parser)
    args, unknown = parser.parse_known_args()
    config = io_util.load_config(args, unknown)
    main_function(config)
