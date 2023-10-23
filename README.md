# Early Bird 🦅

**EarlyBird: Early-Fusion for Multi-View Tracking in the Bird's Eye View**

Torben Teepe, Philipp Wolters, Johannes Gilg, Fabian Herzog, Gerhard Rigoll

[![arxiv](https://img.shields.io/badge/arXiv-2310.13350-red)](https://arxiv.org/abs/2310.13350)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/earlybird-early-fusion-for-multi-view/multi-object-tracking-on-wildtrack)](https://paperswithcode.com/sota/multi-object-tracking-on-wildtrack?p=earlybird-early-fusion-for-multi-view)

## Usage

### Getting Started
1. Install [PyTorch](https://pytorch.org/get-started/locally/) with CUDA support
    ```shell
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
2. Install [mmcv](https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip) with CUDA support
   ```shell
   pip install mmcv==2.0.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
   ```
3. Install remaining dependencies
   ```shell
   pip install -r requirements.txt
   ```

#### Training
```shell
python main.py fit -c configs/t_fit.yml \
    -c configs/d_{multiviewx,wildtrack}.yml
```

#### Testing
```shell
python main.py test -c model_weights/config.yaml \
    --ckpt model_weights/model-epoch=35-val_loss=6.50.ckpt
```

## Acknowledgement
- [Simple-BEV](https://simple-bev.github.io): Adam W. Harley
- [MVDeTr](https://github.com/hou-yz/MVDeTr): Yunzhong Hou

## Cite
If you use EarlyBird, please use the following BibTeX entry.

```
@misc{teepe2023earlybird,
      title={Early{B}ird: Early-Fusion for Multi-View Tracking in the Bird's Eye View}, 
      author={Torben Teepe and Philipp Wolters and Johannes Gilg and Fabian Herzog and Gerhard Rigoll},
      year={2023},
      eprint={2310.13350},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
