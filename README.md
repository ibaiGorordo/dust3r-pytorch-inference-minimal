# dust3r-pytorch-inference-minimal
Minimal code for running the Dust3r model with PyTorch.
![dust3r](https://github.com/user-attachments/assets/2718f715-f23a-4261-8ef8-d8063a9ba0f0)

## Why
The goal of this repository is to provide the minimal code to test and understand how the model works.
- The official model contains a lot of code that is not necessary for inference.
- The model is relatively simple, however handling more than 2 images requires global alignment.
- To make the model easier to be exported to ONNX.

## Main Changes
- Assume image size is constant for all images
- Precalculate rope2d's sines and cosines since the image size is constant
- Divide the model into encoder, decoder and head
- Replace `expm1` in the depth estimation head with `exp -1` to make it possible to export to ONNX
- Add visualization with rerun-sdk
- Remove unnecessary code for the minimum of running the model with 2 images

## Limitations
Not all features are supported, such as global alignment, more than 2 images, etc. For those cases, the following repository is **recommended**:
- [mini-duster](https://github.com/pablovela5620/mini-dust3r): A miniature version of dust3r only for performing inference. This makes it much easier to use without needing the training/data/eval code. Tested on Linux, Apple Silicon Macs, and Windows

## Installation

### Option1: pip installation
```bash
git clone https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal.git
cd dust3r-pytorch-inference-minimal
pip install -r requirements.txt
```

### Option2: uv installation
Recommended but requires uv to be installed (https://docs.astral.sh/uv/getting-started/installation/)
```bash
git clone https://github.com/ibaiGorordo/dust3r-pytorch-inference-minimal.git
cd dust3r-pytorch-inference-minimal
uv sync
```

- Additionally, to activate the uv environment: `source .venv/bin/activate` in macOS and Linux or `.venv\Scripts\activate` in Windows

## Examples:

### Default example (visualize colored pointcloud):
You can additionally set `visualize_depth=True` when using `visualize_output()` function to visualize the depth map.
```bash
python example_image_pair.py
```

### Raw output example:
Example to visualize the raw output of the model: the pointclouds from both cameras aligned with the first camera.
```bash
python example_raw_output.py
```
![dust3r_raw](https://github.com/user-attachments/assets/9fea7e82-e5c6-47a8-ad1c-cf5f609b668a)

## ONNX Export
The model is divided into the encoder (dynamic batch size) and decoder-head when exporting into ONNX.

```bash
python export_onnx.py
```

## License
The code is taken from the official Dust3r repository which is distributed under the CC BY-NC-SA 4.0 License.
See [LICENSE](https://github.com/naver/dust3r/blob/main/LICENSE) for more information.

```python
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
```

### References:
- dust3r: https://github.com/naver/dust3r
- mini-dust3r: https://github.com/pablovela5620/mini-dust3r
- huggingface hub dust3r checkpoints: https://huggingface.co/camenduru/dust3r/tree/main
