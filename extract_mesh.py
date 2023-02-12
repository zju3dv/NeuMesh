import os
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import mcubes
import open3d as o3d

import torch

from models.frameworks import build_framework
from utils import io_util
from utils.checkpoints import load_ckpt

torch.backends.cudnn.benchmark = True


def get_opts():
    parser = io_util.create_args_parser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--chunk",
        type=int,
        default=2 * 1024,
        help="chunk size to split the input to avoid OOM",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="pretrained checkpoint path to load",
    )
    parser.add_argument(
        "--use_emb_a",
        default=False,
        action="store_true",
        help="appearance embedding",
    )
    parser.add_argument(
        "--N_grid",
        type=int,
        # default=512,
        default=256,
        help="size of the grid on 1 side, larger=higher resolution",
    )
    parser.add_argument(
        "--x_range",
        nargs="+",
        type=float,
        default=[-1.0, 1.0],
        help="x range of the object",
    )
    parser.add_argument(
        "--y_range",
        nargs="+",
        type=float,
        default=[-1.0, 1.0],
        help="x range of the object",
    )
    parser.add_argument(
        "--z_range",
        nargs="+",
        type=float,
        default=[-1.0, 1.0],
        help="x range of the object",
    )
    parser.add_argument(
        "--sdf_th",
        type=float,
        default=0.0,
        help="threshold to consider a location is occupied",
    )
    parser.add_argument("--obj_id", type=str, default="0", help="obj_id")

    parser.add_argument("--scale_factor", type=float, default=1.0, help="scale")

    parser.add_argument("--select_pt", nargs="+", type=float, default=None)

    return parser.parse_known_args()


def write_json(content, fname):
    with open(fname, "wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def map_to_color(x, cmap="coolwarm", vmin=None, vmax=None):
    if vmin == None or vmax == None:
        vmin = min(x)
        vmax = max(x)
    colors = plt.cm.get_cmap(cmap)((x - vmin) / (vmax - vmin))[:, :3]
    return colors


def extract_mesh(
    model,
    N_grid,
    x_range,
    y_range,
    z_range,
    sdf_th,
    chunk,
    scale_factor,
    output_dir,
    obj_id,
):
    # define the dense grid for query
    N = N_grid
    xmin, xmax = x_range
    ymin, ymax = y_range
    zmin, zmax = z_range
    # assert xmax-xmin == ymax-ymin == zmax-zmin, 'the ranges must have the same length!'
    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    z = np.linspace(zmin, zmax, N)

    xyz_ = np.stack(np.meshgrid(x, y, z), -1).reshape(-1, 3).astype(np.float32)
    xyz_ = torch.FloatTensor(xyz_).cuda()
    dir_ = torch.zeros_like(xyz_).cuda()
    # sigma is independent of direction, so any value here will produce the same result

    obj_id = obj_id

    # predict sigma (occupancy) for each grid location
    print("Predicting occupancy ...")
    with torch.no_grad():
        B = xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, chunk)):
            xyz_chunk = xyz_[i : i + chunk]  # (N, 3)
            dir_chunk = dir_[i : i + chunk]  # (N, 3)
            res_chunk, _ = model(xyz_chunk, dir_chunk)
            out_chunks += [res_chunk.cpu()]
        sdf = torch.cat(out_chunks, 0)

    sdf = sdf.numpy().reshape(N, N, N)

    print("Extracting mesh ...")
    vertices, triangles = mcubes.marching_cubes(sdf, sdf_th)
    vertices_ = (vertices / N).astype(np.float64)
    x_ = (ymax - ymin) * vertices_[:, 1] + ymin
    y_ = (xmax - xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax - zmin) * vertices_[:, 2] + zmin

    print("Predicting color ...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(
        vertices_.astype(np.float64) * scale_factor
    )
    mesh.triangles = o3d.utility.Vector3iVector(triangles.astype(np.int32))

    mesh.compute_vertex_normals()
    vertices = torch.FloatTensor(vertices_).cuda()
    rays_d = -1 * torch.FloatTensor(np.asarray(mesh.vertex_normals)).cuda()
    with torch.no_grad():
        B = vertices.shape[0]
        out_color_chunks = []
        for i in tqdm(range(0, B, chunk)):
            xyz_chunk = vertices[i : i + chunk]  # (N, 3)
            dir_chunk = rays_d[i : i + chunk]
            _, res_chunk = model(xyz_chunk, dir_chunk)
            out_color_chunks += [res_chunk.cpu()]
        colors = torch.cat(out_color_chunks, 0)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors.numpy().astype(np.float64))

    o3d.io.write_triangle_mesh(
        os.path.join(output_dir, f"extracted_{obj_id}.ply"), mesh
    )

    bbox = mesh.get_axis_aligned_bounding_box()
    bound = np.array([bbox.min_bound, bbox.max_bound])
    size = bound[1] - bound[0]
    write_json(
        {
            "max_bound": bbox.max_bound.tolist(),
            "min_bound": bbox.min_bound.tolist(),
            "size": size.tolist(),
        },
        os.path.join(output_dir, f"bbox_{obj_id}.json"),
    )
    print(bbox)


if __name__ == "__main__":
    args, unknown = get_opts()
    os.makedirs(args.output_dir, exist_ok=True)
    config = io_util.load_config(args, unknown)

    conf = {
        "inside_out": args.obj_id == "0",
        "model": {
            "N_max_objs": 128,
            "N_obj_embedding": 64,
        },
    }

    conf["model"].update({"N_max_lights": 1024, "N_light_embedding": 16})
    if args.use_emb_a:
        conf["model"].update(
            {"N_max_appearance_frames": 10000, "N_appearance_embedding": 16}
        )

    model, trainer, _, _, _ = build_framework(config, config.model.framework)
    model.cuda().eval()
    load_ckpt(args.ckpt_path, model)

    extract_mesh(
        model,
        args.N_grid,
        args.x_range,
        args.y_range,
        args.z_range,
        args.sdf_th,
        args.chunk,
        args.scale_factor,
        args.output_dir,
        args.obj_id,
    )
