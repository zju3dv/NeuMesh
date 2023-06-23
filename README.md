# NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing

### [Project Page](https://zju3dv.github.io/neumesh/) | [Video](https://www.youtube.com/watch?v=8Td3Oy7y_Sc) | [Paper](http://www.cad.zju.edu.cn/home/gfzhang/papers/neumesh/neumesh.pdf)
<div align=center>
<img src="assets/teaser.gif" width="100%"/>
</div>

> [NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing](http://www.cad.zju.edu.cn/home/gfzhang/papers/neumesh/neumesh.pdf)  
> 
> [[Bangbang Yang](https://ybbbbt.com), [Chong Bao](https://github.com/1612190130/)]<sup>Co-Authors</sup>, [Junyi Zeng](https://github.com/LangHiKi/), [Hujun Bao](http://www.cad.zju.edu.cn/home/bao/), [Yinda Zhang](https://www.zhangyinda.com/), [Zhaopeng Cui](https://zhpcui.github.io/), [Guofeng Zhang](http://www.cad.zju.edu.cn/home/gfzhang/). 
> 
> ECCV 2022 Oral
> 


<!-- ⚠️ Note: This is only a preview version of the code. Full code (with training scripts) will be released soon. -->

## Installation
We have tested the code on Python 3.8.0 and PyTorch 1.8.1, while a newer version of pytorch should also work.
The steps of installation are as follows:

* create virtual environmental: `conda env create --file environment.yml`
* install pytorch 1.8.1: `pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111  -f https://download.pytorch.org/whl/torch_stable.html`
* install [open3d **development**](http://www.open3d.org/docs/latest/getting_started.html) version: `pip install [open3d development package url]`
* install [FRNN](https://github.com/lxxue/FRNN), a fixed radius nearest neighbors search implemented on CUDA.

## Data
We use DTU data of [NeuS version](https://github.com/Totoro97/NeuS) and [NeRF synthetic data](https://zjueducn-my.sharepoint.com/:u:/g/personal/12021089_zju_edu_cn/EUTdA3riYNFAiCVas_f7ByEBfFykG_ow9fZy_fD37drgqg?e=RgyEAK).
<!-- Our code reads the poses following the format of `camera_sphere.npz`. 
Therefore, we convert the poses of NeRF synthetic data to [`camera_sphere.npz`](). -->

P.S. Please enable the `intrinsic_from_cammat: True` for `hotdog`, `chair`, `mic` if you use the provided NeRF synthetic dataset.

## Train
Here we show how to run our code on one example scene.
Note that the `data_dir` should be specified in the `configs/*.yaml`.

1. Train the teacher network (NeuS) from multi-view images.
```python
python train.py --config configs/neus_dtu_scan63.yaml
```
2. Extract a triangle mesh from a trained teacher network.
```python
python extract_mesh.py --config configs/neus_dtu_scan63.yaml --ckpt_path logs/neus_dtuscan63/ckpts/latest.pt --output_dir out/neus_dtuscan63/mesh
```
3. Train NeuMesh from multi-view images and the teacher network. Note that the `prior_mesh`, `teacher_ckpt`, `teacher_config` should be specified in the `neumesh*.yaml`
```python
python train.py --config configs/neumesh_dtu_scan63.yaml
```
## Evaluation
<!-- Here we provide a [pre-trained model](https://zjueducn-my.sharepoint.com/:f:/g/personal/12021089_zju_edu_cn/EgCdXYjaVThOnlxq5Xy-2RcBBgOSmwSxMJRtaLlSPp_mlQ?e=268T7p) of DTU scan 63. -->

Here we provide all [pre-trained models](https://zjueducn-my.sharepoint.com/:f:/g/personal/12021089_zju_edu_cn/EmEjhcYEOGZJj_EOOMYPFxQBkzBGzwYFyYPaWeob0PTSng?e=Nf1Lg1) of DTU and NeRF synthetic dataset.

You can evaluate images with the trained models. 

```python
python -m render --config configs/neumesh_dtu_scan63.yaml   --load_pt logs/neumesh_dtuscan63/ckpts/latest.pt --camera_path spiral --background 1 --test_frame 24 --spiral_rad 1.2
```
P.S. If the time of inference costs too much, `--downscale` can be enabled for acceleration.


<div align=center>
<img src="assets/nvs.gif" width="60%"/>
</div>

## Manipulation
Please refer to [`editing/README.md`](editing/README.md).





## Citing
```
@inproceedings{neumesh,
    title={NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing},
    author={{Chong Bao and Bangbang Yang} and Zeng Junyi and Bao Hujun and Zhang Yinda and Cui Zhaopeng and Zhang Guofeng},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```
Note: joint first-authorship is not really supported in BibTex; you may need to modify the above if not using CVPR's format. For the SIGGRAPH (or ACM) format you can try the following:
```
@inproceedings{neumesh,
    title={NeuMesh: Learning Disentangled Neural Mesh-based Implicit Field for Geometry and Texture Editing},
    author={{Bao and Yang} and Zeng Junyi and Bao Hujun and Zhang Yinda and Cui Zhaopeng and Zhang Guofeng},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2022}
}
```
## Acknowledgement
In this project we use parts of the implementations of the following works:

* [NeuS](https://github.com/Totoro97/NeuS) by Peng Wang
* [neurecon](https://github.com/ventusff/neurecon) by ventusff

We thank the respective authors for open sourcing their methods.





