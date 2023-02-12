import open3d as o3d
import numpy as np
import abc

import torch

from utils import io_util
from utils.vis_mesh_util import vis_and_painting

from models.frameworks.neumesh import get_model
from models.renderer import SingleRenderer
from .editable_primitive import EditablePrimitive
from .texture_neumesh import TextureEditableNeuMesh

from render import render_function
from editing.render_geometry_editing import deform_model


class TextureEditableRenderer(abc.ABC):
    def __init__(self):
        pass

    def forward(self, args):
        device = args.device

        # read data
        main_primitive, main_args, render_kwargs_test = self.read_data(
            args.main_config,
            args.main_mask_mesh,
            args.main_ckpt,
            device,
            args.debug_draw,
        )
        ref_primitives = []
        for i in range(len(args.ref_config)):
            ref_primitive, _, _ = self.read_data(
                args.ref_config[i],
                [args.ref_mask_mesh[i]],
                args.ref_ckpt[i],
                device,
                args.debug_draw,
            )
            ref_primitives.append(ref_primitive)
        assert main_primitive.get_len_of_mask() == len(
            ref_primitives
        ), "Error: the number of main mask is not mached with number of ref objects"

        # texture edit
        T_r_m_list = self.transfer_texture_features(
            args, main_primitive, ref_primitives
        )

        # create TextureEditableMBGS
        print("[Info] create TextureEditableMBGS")
        main_model = main_primitive.get_model()
        main_editing_masks = main_primitive.get_editing_masks(True)
        main_editing_colorfeats = main_primitive.get_color_features()
        ref_models = []
        for i in range(len(ref_primitives)):
            ref_models.append(ref_primitives[i].get_model())
        if T_r_m_list is not None:
            T_r_m_list = torch.FloatTensor(T_r_m_list)
        model = TextureEditableNeuMesh(
            main_model,
            ref_models,
            main_editing_masks,
            main_editing_colorfeats,
            T_r_m_list,
        )
        model.to(device).eval()

        # render_view
        renderer = SingleRenderer(model)
        args.update(dict(main_args))
        render_function(args, render_kwargs_test, renderer)

    def read_data(self, main_config, mask_paths, ckpt_file, device, debug_draw):
        # create main model
        main_args = io_util.load_yaml(main_config)
        (
            model,
            _,
            _,
            render_kwargs_test,
            _,
        ) = get_model(main_args)
        state_dict = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(state_dict["model"])

        editing_params_list = []
        for mask_path in mask_paths:
            editing_params = self.read_editing_mask(mask_path, model.mesh_grid.mesh)
            editing_params_list.append(editing_params)

        primitive = EditablePrimitive(
            model,
            editing_params_list,
            color_feature_ini=torch.zeros_like(model.color_features),
        )

        if debug_draw == True:
            print("[Info] visualize mesh with editing mask")
            painting_mask = (
                np.zeros(editing_params_list[0].get_editing_mask().shape) == 1
            )
            for editing_params in editing_params_list:
                painting_mask = painting_mask | editing_params.get_editing_mask()
            vis_and_painting(model.mesh_grid.mesh, painting_mask)

        return (
            primitive,
            main_args,
            render_kwargs_test,
        )

    @abc.abstractmethod
    def read_editing_mask(self, mask_path, mesh):
        raise NotImplementedError

    @abc.abstractmethod
    def transfer_texture_features(
        self,
        args,
        main_primitive,
        ref_primitives,
    ):
        raise NotImplementedError
