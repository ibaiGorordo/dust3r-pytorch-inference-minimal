import numpy as np
import cv2
import torch

import torchvision.transforms as T


ImgNorm = T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def preprocess(img: np.ndarray, width: int, height: int, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Crop resize
    original_height, original_width = frame.shape[:2]
    if original_height / height < original_width / width:
        scale = height / original_height
    else:
        scale = width / original_width
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    frame = frame[(new_height - height) // 2:(new_height + height) // 2,
                  (new_width - width) // 2:(new_width + width) // 2, :]

    return ImgNorm(frame).unsqueeze(0).to(device), frame
