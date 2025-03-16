from dataclasses import dataclass
from enum import Enum
import os
import requests
from tqdm import tqdm

from huggingface_hub import hf_hub_url
import numpy as np
import torch

model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

class ModelType(Enum):
    DUSt3R_ViTLarge_BaseDecoder_224_linear = "DUSt3R_ViTLarge_BaseDecoder_224_linear"
    DUSt3R_ViTLarge_BaseDecoder_512_linear = "DUSt3R_ViTLarge_BaseDecoder_512_linear"
    DUSt3R_ViTLarge_BaseDecoder_512_dpt = "DUSt3R_ViTLarge_BaseDecoder_512_dpt"

def download(url: str, filename: str):
    with open(filename, 'wb') as f:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total = int(r.headers.get('content-length', 0))

            # tqdm has many interesting parameters. Feel free to experiment!
            tqdm_params = {
                'total': total,
                'miniters': 1,
                'unit': 'B',
                'unit_scale': True,
                'unit_divisor': 1024,
            }
            with tqdm(**tqdm_params) as pb:
                for chunk in r.iter_content(chunk_size=8192):
                    pb.update(len(chunk))
                    f.write(chunk)


def download_hf_model(filename: str, model_dir: str = model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    filename = filename.split('.')[0] + '.pth'
    path = model_dir + "/" + filename
    if os.path.exists(path):
        return path

    print(f"Model {filename} not found, downloading from Hugging Face Hub...")

    repo_id = "camenduru/dust3r"

    url = hf_hub_url(repo_id=repo_id, filename=filename)
    download(url, path)
    print("Model downloaded successfully to", path)

    return path


def get_device() -> torch.device:
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')

    return device

def calculate_img_size(img_size: tuple[int, int],
                       input_size: int,
                       pad_size: int = 16) -> tuple[int, int]:

    # If the input size is 224, return directly 224, 224
    if input_size == 224:
        return 224, 224

    # Calculate the new image size to keep the aspect ratio but being multiple of pad_size
    w, h = img_size
    # Checl the aspect ratio
    if h > w:
        new_h = input_size
        new_w = int(w * new_h / h)
    else:
        new_w = input_size
        new_h = int(h * new_w / w)
    # Make the new image size multiple of pad_size
    new_h = int(np.ceil(new_h / pad_size) * pad_size)
    new_w = int(np.ceil(new_w / pad_size) * pad_size)
    return new_w, new_h


@dataclass
class Output:
    input: np.ndarray
    pts3d: np.ndarray
    colors: np.ndarray
    conf_map: np.ndarray
    depth_map: np.ndarray
    intrinsic: np.ndarray
    pose: np.ndarray
    width: int
    height: int
