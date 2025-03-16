# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
#  Main Dust3D class
# --------------------------------------------------------
import os

import numpy as np
import torch
import torch.nn as nn

from .utils import Output, ModelType, download_hf_model
from .encoder import Dust3rEncoder
from .decoder import Dust3rDecoder
from .head import Dust3rHead
from .preprocess import preprocess
from .postprocess import postprocess_symmetric, postprocess

model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')

class Dust3r(nn.Module):
    def __init__(self,
                 model_type: ModelType,
                 width: int = 512,
                 height: int = 512,
                 encoder_batch_size: int = 2,
                 symmetric: bool = False,
                 device: torch.device = torch.device('cuda'),
                 conf_threshold: float = 3.0,
                 ):
        super().__init__()

        self.width = width
        self.height = height
        self.symmetric = symmetric
        self.device = device
        self.conf_threshold = conf_threshold

        model_path = download_hf_model(model_type.value)
        ckpt_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        self.encoder = Dust3rEncoder(ckpt_dict, width=width, height=height, device=device, batch=encoder_batch_size)
        self.decoder = Dust3rDecoder(ckpt_dict, width=width, height=height, device=device)
        self.head = Dust3rHead(ckpt_dict, width=width, height=height, device=device)

    def __call__(self, img1: np.ndarray, img2: np.ndarray) -> tuple[Output, Output]:
        return self.forward(img1, img2)

    @torch.inference_mode()
    def forward(self, img1: np.ndarray, img2: np.ndarray) -> tuple[Output, Output]:

        input1, frame1 = preprocess(img1, self.width, self.height, self.device)
        input2, frame2 = preprocess(img2, self.width, self.height, self.device)

        input = torch.cat((input1, input2), dim=0)
        feat = self.encoder(input)
        feat1, feat2 = feat.chunk(2, dim=0)

        pt1_1, cf1_1, pt2_1, cf2_1 = self.decoder_head(feat1, feat2)
        if self.symmetric:
            pt2_2, cf2_2, pt1_2, cf1_2 = self.decoder_head(feat2, feat1)

            output1, output2 = postprocess_symmetric(frame1, pt1_1, cf1_1, pt1_2, cf1_2,
                                                     frame2, pt2_1, cf2_1, pt2_2, cf2_2,
                                                     self.conf_threshold, self.width, self.height,
                                                     )
        else:
            output1, output2 = postprocess(frame1, pt1_1, cf1_1,
                                           frame2, pt2_1, cf2_1,
                                           self.conf_threshold, self.width, self.height,
                                           )


        return output1, output2

    def decoder_head(self, feat1, feat2):
        d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12 = self.decoder(feat1, feat2)
        pt1, cf1, pt2, cf2 = self.head(d1_0, d1_6, d1_9, d1_12, d2_0, d2_6, d2_9, d2_12)
        return pt1, cf1, pt2, cf2
