import os
import torch
from utils import rend_util
from utils.io_util import load_mask, glob_imgs


class SceneDataset(torch.utils.data.Dataset):
    def __init__(self, img_dataset):
        images = img_dataset.get_images()
        masks = img_dataset.get_masks()
        intrinsics = img_dataset.get_intrinsics()
        c2ws = img_dataset.get_c2ws()
        self.H, self.W = img_dataset.get_image_size()

        paint_mask_dir = os.path.join(img_dataset.instance_dir, "paint_mask")
        paint_mask_paths = sorted(glob_imgs(paint_mask_dir))
        self.paint_mask = []
        for i, path in enumerate(paint_mask_paths):
            paint_mask = load_mask(path, img_dataset.downscale)
            paint_mask = paint_mask.reshape(-1)
            self.paint_mask.append(torch.from_numpy(paint_mask).to(dtype=torch.bool))

        assert (
            len(images)
            == len(masks)
            == len(intrinsics)
            == len(c2ws)
            == len(self.paint_mask)
        )

        rays_d_paint = []
        rays_o_paint = []
        mask_paint = []
        rgb_paint = []
        rays_o_bg = []
        rays_d_bg = []
        rgb_bg = []
        mask_bg = []
        for i in range(len(images)):
            c2w = c2ws[i]
            intrinsic = intrinsics[i]
            image = images[i]
            paint_mask = self.paint_mask[i]
            img_mask = masks[i]
            img_mask[paint_mask] = False
            rays_o, rays_d, select_inds = rend_util.get_rays(
                c2w, intrinsic, self.H, self.W, N_rays=-1
            )
            target_rgb = torch.gather(image, 0, torch.stack(3 * [select_inds], -1))
            selected_paint_mask = torch.gather(paint_mask, 0, select_inds)
            selected_img_mask = torch.gather(img_mask, 0, select_inds)

            rays_o_paint.append(rays_o[selected_paint_mask])
            rays_d_paint.append(rays_d[selected_paint_mask])
            rgb_paint.append(target_rgb[selected_paint_mask])
            mask_paint.append(torch.ones(selected_paint_mask.shape) == 1)

            rays_o_bg.append(rays_o[selected_img_mask])
            rays_d_bg.append(rays_d[selected_img_mask])
            rgb_bg.append(target_rgb[selected_img_mask])
            mask_bg.append(torch.ones(selected_img_mask.shape) == 1)

        self.rays_o_paint = torch.cat(rays_o_paint, dim=0)
        self.rays_d_paint = torch.cat(rays_d_paint, dim=0)
        self.mask_paint = torch.cat(mask_paint, dim=0)
        self.rgb_paint = torch.cat(rgb_paint, dim=0)
        self.num_paint = len(self.rgb_paint)

        self.rays_o_bg = torch.cat(rays_o_bg, dim=0)
        self.rays_d_bg = torch.cat(rays_d_bg, dim=0)
        self.mask_bg = torch.cat(mask_bg, dim=0)
        self.rgb_bg = torch.cat(rgb_bg, dim=0)
        self.num_bg = len(self.rgb_bg)

    def __len__(self):
        return max(self.num_paint, self.num_bg)

    def __getitem__(self, idx):

        idx_paint = idx % self.num_paint
        idx_bg = idx % self.num_bg

        sample = {
            "rays_o_paint": self.rays_o_paint[idx_paint],
            "rays_d_paint": self.rays_d_paint[idx_paint],
            "mask_paint": self.mask_paint[idx_paint],
            "rays_o_bg": self.rays_o_bg[idx_bg],
            "rays_d_bg": self.rays_d_bg[idx_bg],
            "mask_bg": self.mask_bg[idx_bg],
        }

        ground_truth = {
            "rgb_paint": self.rgb_paint[idx_paint],
            "rgb_bg": self.rgb_bg[idx_bg],
        }

        return idx, sample, ground_truth
