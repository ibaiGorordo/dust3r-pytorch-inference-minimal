import onnx
import torch
import torch.nn as nn
from dust3r import Dust3r, ModelType
from onnxsim import simplify


class Dust3rDecoderHead(nn.Module):
    def __init__(self, model: Dust3r):
        super().__init__()
        self.model = model

    def forward(self, f1, f2):
        f1_0, f1_6, f1_9, f1_12, f2_0, f2_6, f2_9, f2_12 = self.model.decoder(f1, f2)
        pts3d1, conf1, pts3d2, conf2 = self.model.head(f1_0, f1_6, f1_9, f1_12, f2_0, f2_6, f2_9, f2_12)
        return pts3d1, conf1, pts3d2, conf2

if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    width, height = 512, 288
    encoder_output_path = "models/dust3r_encoder.onnx"
    decoder_output_path = "models/dust3r_decoder_head.onnx"
    model_type = ModelType.DUSt3R_ViTLarge_BaseDecoder_512_dpt
    dust3r = Dust3r(model_type, width, height, device=device)
    decoder_head = Dust3rDecoderHead(dust3r).to(device) # Combined decoder and head

    img1 = torch.randn(1, 3, height, width).to(device)
    img2 = torch.randn(1, 3, height, width).to(device)
    feat = dust3r.encoder(torch.cat((img1, img2)))
    f1, f2 = feat.chunk(2)

    print("Exporting encoder...")
    torch.onnx.export(
        dust3r.encoder,
        (torch.cat((img1, img2)),),
        encoder_output_path,
        input_names=["imgs"],
        output_names=["feats"],
        dynamic_axes={"imgs": {0: "batch"}, "feats": {0: "batch"}},
        opset_version=13,
    )

    print("Simplifying encoder...")
    encoder_onnx = onnx.load(encoder_output_path)
    simplified_encoder_onnx, _ = simplify(encoder_onnx)
    onnx.save(simplified_encoder_onnx, encoder_output_path)

    print("Exporting decoder head...")
    torch.onnx.export(
        decoder_head,
        (f1, f2),
        decoder_output_path,
        input_names=["feat1", "feat2"],
        output_names=["pts3d1", "conf1", "pts3d2", "conf2"],
        opset_version=13,
    )

    print("Simplifying decoder head...")
    decoder_head_onnx = onnx.load(decoder_output_path)
    simplified_decoder_head_onnx, _ = simplify(decoder_head_onnx)
    onnx.save(simplified_decoder_head_onnx, decoder_output_path)