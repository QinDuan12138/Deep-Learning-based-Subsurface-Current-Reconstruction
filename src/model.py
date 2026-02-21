import logging
import torch
import torch.nn as nn

from swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)

class SwinUnet(nn.Module):
    def __init__(self, config, img_size=512, num_classes=44, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=self.num_classes,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT
        )

    def forward(self, x):

        logits = self.swin_unet(x)
        return logits

if __name__ == "__main__":
    
    from config import _C as config  # Import default configuration
    from thop import profile

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Create model
    model = SwinUnet(config=config, img_size=512, num_classes=44)
    model.to(device)  # Move model to GPU
    model.eval()  # Evaluation mode

    # Create dummy input (batch, 512, 512, 7)
    fake_input = torch.randn(config.DATA.BATCH_SIZE, 512, 512, 7).to(device)
    fake_input = fake_input.permute(0, 3, 1, 2)  # (B, C, H, W)

    # Forward pass
    with torch.no_grad():
        output = model(fake_input)

    print("\nOutput shape:", output.shape)
    print("Output device:", output.device)

     # Use thop to compute FLOPs and parameter count
    flops, params = profile(model, inputs=(fake_input,))
    print(f"FLOPs: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")