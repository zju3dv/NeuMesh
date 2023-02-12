import numpy as np


def get_data(args, return_val=False, val_downscale=4.0, **overwrite_cfgs):
    dataset_type = args.data.get("type", "DTU")
    cfgs = {
        "scale_radius": args.data.get("scale_radius", -1),
        "downscale": args.data.downscale,
        "data_dir": args.data.data_dir,
        "train_cameras": False,
        "split": args.data.get("split", "entire"),
    }
    # assert cfgs["split"] is not None

    if dataset_type == "DTU":
        from .DTU import SceneDataset

        cfgs["intrinsic_from_cammat"] = args.data.get("intrinsic_from_cammat", False)
        cfgs["cam_file"] = args.data.get("cam_file", None)
    else:
        raise NotImplementedError

    cfgs.update(overwrite_cfgs)
    paint_dataset = args.data.get("paint_dataset", False)
    if return_val:
        if cfgs["split"] == "entire":
            dataset = SceneDataset(**cfgs)
            cfgs["downscale"] = val_downscale
            val_dataset = SceneDataset(**cfgs)
            return dataset, val_dataset
        else:
            cfgs["split"] = "train"
            dataset = SceneDataset(**cfgs)
            cfgs["downscale"] = val_downscale
            cfgs["split"] = "val"
            val_dataset = SceneDataset(**cfgs)
            return dataset, val_dataset
    else:
        dataset = SceneDataset(**cfgs)
        if paint_dataset == False:
            return dataset
        else:
            from .paint_dataset import SceneDataset as PaintDataset

            return PaintDataset(dataset)
