# ml_mdm - Matryoshka Diffusion Models

`ml_mdm` is a python package for efficiently training high quality text-to-image diffusion models — brought to the public by [Luke Carlson](https://github.com/luke-carlson), [Jiatao Gu](https://github.com/MultiPath), [Shuangfei Zhai](https://github.com/Shuangfei), and [Navdeep Jaitly](https://github.com/ndjaitly).


---


<div align="center">


This software project accompanies the research paper, [*Matryoshka Diffusion Models*](https://arxiv.org/abs/2310.15111).


*Jiatao Gu, Shuangfei Zhai, Yizhe Zhang, Josh Susskind, Navdeep Jaitly*

[[`Paper`](https://arxiv.org/abs/2310.15111)]  [[`BibTex`](#citation)]




![mdm text to image outputs](https://mlr.cdn-apple.com/media/MDM_text_to_image_390ce54fde.png)

</div>


## Table of Contents

| Section | Description |
| - | - |
| [Introduction](#introduction) | A brief overview of Matryoshka Diffusion Models |
| [Installation](#installation) | Start training models and generating samples with `ml_mdm` |
| [Pretrained Models](#pretrained-models) | Links to download our pretrained models (64, 256, 1024) |
| [Web Demo](#web-demo) | Generate images with our web UI |
| [Codebase Structure](#codebase) | An overview of the python module |
| [Concepts](#concepts) | Core concepts and design principles. |
| [Tutorial](#tutorials) | Step-by-step training of an MDM model on CC12m |




## Installation
The default installation dependencies, as defined in the `pyproject.toml`, are selected so that you can install this library even on a CPU only machine.

> Users have run this codebase with Python 3.9,3.10 and cuda_12, cuda-11.8

```
> pip install -e .
```

Developers should set up `pre-commit` as well with `pre-commit install`.

### Running Test Cases

```
> pytest   # will run all test cases - including ones that require a gpu
> pytest  -m "not gpu"  # run test cases that can work with just cpu
```


# Pretrained Models
We've uploaded model checkpoints to:
- https://docs-assets.developer.apple.com/ml-research/models/mdm/flickr64/vis_model.pth
- https://docs-assets.developer.apple.com/ml-research/models/mdm/flickr256/vis_model.pth
- https://docs-assets.developer.apple.com/ml-research/models/mdm/flickr1024/vis_model.pth

> Note: We are releasing models that were trained on 50M text-image pairs collected from Flickr. In this repo, we provide scripts for downloading [CC12M](https://github.com/google-research-datasets/conceptual-12m) and configs for training equivalent models on CC12M data.

Feel free to download the models or skip further down to train your own. Once a pretrained model is downloaded locally, you can use it in our web demo, pass it as an argument to training, sampling, and more.

```console
export ASSET_PATH=https://docs-assets.developer.apple.com/ml-research/models/mdm

curl $ASSET_PATH/flickr64/vis_model.pth --output vis_model_64x64.pth
curl $ASSET_PATH/flickr256/vis_model.pth --output vis_model_256x256.pth
curl $ASSET_PATH/flickr1024/vis_model.pth --output vis_model_1024x1024.pth
```


### Web Demo
You can run your own instance of the web demo (after downloading the checkpoints) with this command:

```console
torchrun --standalone --nproc_per_node=1  ml_mdm/clis/generate_sample.py --port $YOUR_PORT
```

![image](docs/web_demo.png)

## Codebase

| module | description |
| - | - |
| `ml_mdm.models` | The core model implementations |
| `ml_mdm.diffusion` | Model pipelines, for example DDPM |
| `ml_mdm.config` | Connects configuration dataclasses with associated models, pipelines, and clis using [simple parsing](https://github.com/lebrice/SimpleParsing/blob/master/README.md) |
| `ml_mdm.clis` | All command line tools in the project, the most relevant being `train_parallel.py` |
| `tests/` | Unit tests and sample training files |







# Tutorials

## Generate Your Own Images With Pretrained Checkpoints

Once you've installed `ml_mdm`, download these checkpoints into the repo's directory.

```console
curl https://docs-assets.developer.apple.com/ml-research/models/mdm/flickr64/vis_model.pth --output vis_model_64x64.pth
curl https://docs-assets.developer.apple.com/ml-research/models/mdm/flickr256/vis_model.pth --output vis_model_256x256.pth
```

The web demo will load each model with a corresponding configuration:
- `vis_model_64x64.pth` will be loaded with the settings from `configs/models/cc12m_64x64.yaml`
- `vis_model_256x256.pth` will be loaded with the settings from `configs/models/cc12m_256x256.yaml`
- `vis_model_1024x1024.pth` will be loaded with the settings from `configs/models/cc12m_1024x1024.yaml`

In the demo, you can change a variety of settings and peek into the internals of the model. Set the port you'd like to use by swapping in `$YOUR_PORT` and then run:

```console
torchrun --standalone --nproc_per_node=1  ml_mdm/clis/generate_sample.py --port $YOUR_PORT
```



## Citation
If you find our work useful, please consider citing us as:
```
@misc{gu2023matryoshkadiffusionmodels,
      title={Matryoshka Diffusion Models},
      author={Jiatao Gu and Shuangfei Zhai and Yizhe Zhang and Josh Susskind and Navdeep Jaitly},
      year={2023},
      eprint={2310.15111},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2310.15111},
}
```
