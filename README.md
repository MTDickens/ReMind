# Teaching Video Generators to Remember: Eliciting Dynamic Memory for Out-of-Sight State Evolution

Official implementation of **Teaching Video Generators to Remember: Eliciting
Dynamic Memory for Out-of-Sight State Evolution**.

## TODO List

- [x] Release the project page
- [x] Release inference code
- [x] Release ReMind-5B checkpoints
- [ ] Release ReMind-1.3B checkpoints
- [ ] Release training code

ReMind elicits dynamic memory in causal video generators so that hidden world
state can continue evolving through camera motion, occlusion, and illumination
changes.

[🌐 Project page](https://remind-applied.github.io/) ·
[📄 arXiv](https://arxiv.org/abs/2605.25333) ·
[🤗 Models](https://huggingface.co/AppliedIntuitionResearch/ReMind) ·
[🤗 Dataset](https://huggingface.co/datasets/AppliedIntuitionResearch/ReMind1M)

This repository contains the inference implementation, model-loading utilities,
and reproducible I2V/V2V presets for the released ReMind checkpoints. Training
code is not included in this public Git repository.

## Setup

```bash
sudo apt update && sudo apt install python3.12 python3.12-venv
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Google Colab only: make `nvidia-smi` work in shell sessions

Google Colab may expose its NVIDIA driver libraries outside the default search
path used by an SSH or terminal shell. If `nvidia-smi` reports that it cannot
find `libnvidia-ml.so`, persist the Colab library paths in `~/.bashrc`:

```bash
COLAB_NVML_EXPORT='export LD_LIBRARY_PATH="/usr/lib64-nvidia:/usr/local/nvidia/lib:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
grep -qxF "$COLAB_NVML_EXPORT" ~/.bashrc || printf '%s\n' "$COLAB_NVML_EXPORT" >> ~/.bashrc
source ~/.bashrc
nvidia-smi
```

This workaround is only for a Google Colab managed runtime. It is not a driver
installation procedure for a local machine or a standalone Docker container.

Download the corresponding official Wan checkpoint into `checkpoints/`; the
expected paths are listed in `configs/model_5b.yaml` and
`configs/model_1p3b.yaml`. ReMind checkpoints and the release dataset are not
stored in Git; download them from the Hugging Face repositories linked above.

```bash
# cd Agent_player/ReMind

hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B

hf download AppliedIntuitionResearch/ReMind \
  ReMind-5B.safetensors \
  ReMind-5b-dmd-ema.safetensors \
  SHA256SUMS \
  --local-dir checkpoints/ReMind-5B
```

## Inference

**Custom inputs and captions:** read
[`docs/inference.md`](docs/inference.md) for the content-only caption contract,
control-specific prompt schedules, bundled camera-trajectory rules, and a
custom-preset checklist.

**Conditioning mode:** the original seven examples below are **all I2V**—each
uses only one input image. Four additional examples are explicitly labeled
**V2V** and condition on a 21-frame video prefix (6 latent frames) without
reading any future frames.

```bash
python inference.py \
  --preset examples/presets/01_latte_occluder_recovery.yaml \
  --model-folder checkpoints/Wan2.2-TI2V-5B \
  --base-checkpoint checkpoints/ReMind-5B/ReMind-5B.safetensors \
  --ema-checkpoint checkpoints/ReMind-5B/ReMind-5b-dmd-ema.safetensors \
  --output outputs/latte_occluder_recovery.mp4
```

### Inference examples

| Preset | Task | Control |
|---|---|---|
| `01_latte_occluder_recovery.yaml` | **I2V** | latte art evolves behind a cardboard-box occluder |
| `02_pancake_occluder_recovery.yaml` | **I2V** | pancake batter evolves behind a cardboard-box occluder |
| `03_whisking_camera_pan.yaml` | **I2V** | whisking under the selected pair-fixed InSpatio camera trajectory |
| `04_cereal_camera_pan.yaml` | **I2V** | granular pouring under the selected pair-fixed InSpatio camera trajectory |
| `05_dog_clean_i2v.yaml` | **I2V** | clean dog motion |
| `06_dry_ice_clean_i2v.yaml` | **I2V** | clean dry-ice/foam evolution |
| `07_flour_clean_i2v.yaml` | **I2V** | clean granular pouring |
| `08_cake_decorating_occluder_v2v.yaml` | **V2V** | syrup decoration evolves on a frosted slice behind a cardboard-box occluder |
| `09_ice_melting_drink_occluder_v2v.yaml` | **V2V** | a flaming, foaming drink evolves behind a cardboard-box occluder |
| `10_pour_wine_beer_camera_pan_v2v.yaml` | **V2V** | a carbonated drink fills under its pair-fixed InSpatio camera trajectory |
| `11_slime_stretch_camera_pan_v2v.yaml` | **V2V** | viscous yellow slime accumulates under its pair-fixed InSpatio camera trajectory |

The V2V set intentionally contains two occluder and two camera-pan cases, with
no clean V2V case and no content overlap with the seven I2V presets. Its
bundled inputs are the first 21 frames of the corresponding source-reference
clips; inference decodes only those prefix frames and pads the remainder
internally before causal VAE encoding.

Inspect the resolved four-step prompt/camera schedule without loading weights:

```bash
python inference.py --preset examples/presets/01_latte_occluder_recovery.yaml --dry-run
```

See [`docs/inference.md`](docs/inference.md) for caption and control
conventions, and [`examples/README.md`](examples/README.md) for the exact page-case seeds,
occluder schedules, pair-fixed camera trajectories, prompts, and input frames.
Official weights are available from the Hugging Face model repository linked
above.

## Acknowledgements

We thank the authors and maintainers of [Wan2.1](https://github.com/Wan-Video/Wan2.1)
and [Wan2.2](https://github.com/Wan-Video/Wan2.2) for the open models and code
that serve as the foundation of ReMind. We are grateful to
[PRoPE](https://www.liruilong.cn/prope/) for projective camera positional
encoding, [LongLive](https://github.com/NVlabs/LongLive) for its open DMD and
autoregressive video-generation work, [Helios](https://github.com/PKU-YuanGroup/Helios)
for the video generator used in our dynamic-data engine,
[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3) for
geometry preprocessing.

We also thank the teams and data providers behind
[InSpatio-World](https://www.inspatio.com/models/world),
[NeoVerse](https://neoverse-4d.github.io/),
[OpenVid-1M](https://github.com/NJU-PCALab/OpenVid-1M),
[SpatialVID](https://github.com/NJU-3DV/SpatialVID),
[DL3DV-10K](https://dl3dv-10k.github.io/DL3DV-10K/),
[Pexels](https://www.pexels.com/), [Kubric](https://github.com/google-research/kubric),
and [PhyCo](https://huggingface.co/datasets/nnsriram97/phyco_kubric) for the
research and data resources used in this work.

## Licenses

Original ReMind code is licensed under Creative Commons
Attribution-NonCommercial 4.0 International (CC BY-NC 4.0). Wan-derived code
remains Apache-2.0 and preserves its upstream notices;
`prope/camera_rope.py` is MIT. Full third-party license
copies are bundled in `LICENSES/`; see `NOTICE` and
`docs/THIRD_PARTY_NOTICES.md`. Model weights and dataset terms are covered
separately: ReMind weights are CC BY-NC 4.0 under
`docs/WEIGHTS_LICENSE.md`, while final dataset terms remain part of release
review.

## Citation

```bibtex
@article{xu2026teaching,
  title={Teaching Video Generators to Remember: Eliciting Dynamic Memory for Out-of-Sight State Evolution},
  author={Xu, Tianshuo and Xie, Yichen and Meng, Depu and Peng, Chensheng and Herau, Quentin and Jiang, Bo and Hu, Yihan and Zhan, Wei},
  journal={arXiv preprint arXiv:2605.25333},
  year={2026}
}
```
