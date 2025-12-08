<p align="center" />
<h1 align="center">MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference</h1>

<p align="center">
    <a href="https://wen-yuan-zhang.github.io/"><strong>Wenyuan Zhang</strong></a>
    ·
    <a href="https://github.com/tangjm24"><strong>Jiming Tang</strong></a>
    ·
    <a href="https://github.com/weiqi-zhang"><strong>Weiqi Zhang</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu†</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>
</p>
<h2 align="center">NeurIPS 2025</h2>
<h3 align="center"><a href="https://arxiv.org/abs/2510.11387">Paper</a> | <a href="https://wen-yuan-zhang.github.io/MaterialRefGS/">Project Page</a></h3>
<div align="center"></div>
<div align="center"></div>

## Video
<video width="780" controls muted loop>
  <source src="assets/video.mp4" type="video/mp4">
</video>



## Pipeline
<p align="center">
    <img src="assets/method.png" width="780" />
</p>

# Setup

## Installation

Clone the repository and create an anaconda environment using
```shell
git clone https://github.com/wen-yuan-zhang/MaterialRefGS
cd MaterialRefGS
conda create -n MaterialRefGS python=3.10
conda activate MaterialRefGS
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt

## install other extensions
pip install submodules/cubemapencoder
pip install submodules/diff-surfel-rasterization
pip install submodules/diff-surfel-rasterization2
pip install submodules/raytracing_brdf
pip install submodules/simple-knn
```
## Dataset & preparation

We use [Metric3D v2](https://github.com/YvanYin/Metric3D) for monocular normal validation. 
In reflective regions, the monocular normal prior helps the Gaussian training process converge faster and constrains the geometry optimization. The `metric3d_path` in `run_*.sh` should be updated to the actual path of the normal prior. You can also modify the `load_normal_prior` function in `train_*.py` to replace it with other normal-prior methods. 

For datasets, we mainly test our method on [Shiny Blender Synthetic](https://storage.googleapis.com/gresearch/refraw360/ref.zip), [Shiny Blender Real](https://storage.googleapis.com/gresearch/refraw360/ref_real.zip) and [Glossy Synthetic](https://liuyuan-pal.github.io/NeRO/). Please run the script `nero2blender.py` to convert the format of the Glossy Synthetic dataset.


# Training

To train refreal dataset, simply run
```shell
chmod +x run_refreal.sh
bash ./run_refreal.sh
```
The other datasets can follow a similar manner to the above. The scene name, experiment name, code/data path are specified in the script. You can adjust them as your own configurations.


# Evaluation

for Shiny Blender Synthetic dataset  and Glossy Synthetic, simple run 
```
python eval.py --white_background --save_images --model_path output_{DATASET}/NAME_OF_THE_SCENE
```
this command will render material maps and calculate the psnr/ssim/lpips metrics. the Shiny Blender Real dataset do not use `--white_background` option.



# Acknowledgements

This project is built upon 
- [Ref-gaussian](https://github.com/fudan-zvg/ref-gaussian)
- [Nero](https://github.com/liuyuan-pal/NeRO)
- [Metric3D v2](https://jugghm.github.io/Metric3Dv2/) 
- [3DGS-DR](https://github.com/gapszju/3DGS-DR)
- [EnvGS](https://github.com/zju3dv/EnvGS). 

We thank all the authors for their great repos.

# Citation

If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{zhangmaterialrefgs,
  title={MaterialRefGS: Reflective Gaussian Splatting with Multi-view Consistent Material Inference},
  author={Zhang, Wenyuan and Tang, Jimin and Zhang, Weiqi and Fang, Yi and Liu, Yu-Shen and Han, Zhizhong},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year={2025}
}
```