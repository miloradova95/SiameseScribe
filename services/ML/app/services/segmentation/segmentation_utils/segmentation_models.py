from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

########################################
### GENERIC MODEL BUILDING FUNCTIONS ###


# Decide which model to build based on the saved model name (for loading saved models)
def build_model_from_name(model_name, num_classes=2):
    """
    Supported prefixes (case-insensitive):
      - UNet-V1_..., UNet-V2_..., UNet-V3_... -> builds corresponding UNet
      - DeepLab-smp... or deeplab* -> builds DeepLab via SMP
      - model_... or maskrcnn... or contains 'maskrcnn' -> builds Mask R-CNN
    If automatic inference fails, raise ValueError.
    """
    model_name_lower = model_name.lower()
    # Semantic segmentation saved model names
    if model_name_lower.startswith("unet-v1_"):
        print("Building UNetV1 model (Baseline custom U-Net)...")
        return UNetV1(in_channels=3, out_channels=num_classes), "semantic"
    elif model_name_lower.startswith("unet-v2_"):
        print("Building UNetV2 model (Attention U-Net)...")
        return UNetV2(in_channels=3, out_channels=num_classes), "semantic"
    elif model_name_lower.startswith("unet-v3_"):
        print("Building UNetV3 model (U-Net++)...")
        return UNetV3(in_channels=3, out_channels=num_classes), "semantic"
    elif model_name_lower.startswith("deeplab-smp_"):
        print("Building DeepLabV3+ model via SMP...")
        return build_deeplab("smp", num_classes=num_classes), "semantic"
    # Mask R-CNN saved model names
    elif (
        model_name_lower.startswith("model_")
        or "maskrcnn" in model_name_lower
        or model_name_lower.startswith("maskrcnn")
    ):
        print("Building Mask R-CNN model...")
        return build_maskrcnn(num_classes), "maskrcnn"
    else:
        raise ValueError(f"Unknown model name prefix: {model_name}. Expected UNet-, DeepLab- or MaskRCNN-like name.")


# Build semantic segmentation model based on architecture and version (for unet_deeplab_train.py)
def build_semantic_model(version, num_classes, architecture="unet", deeplab_variant="torch"):
    """
    architecture: 'unet' (default) or 'deeplab'
    version: for unet -> '1','2','3' etc. (for deeplab this is ignored)
    deeplab_variant: 'smp' (default) when architecture=='deeplab'
    """
    architecture = architecture.lower()
    if architecture == "unet":
        if version == "1":
            return UNetV1(in_channels=3, out_channels=num_classes)
        elif version == "2":
            return UNetV2(in_channels=3, out_channels=num_classes)
        elif version == "3":
            return UNetV3(in_channels=3, out_channels=num_classes)
        else:
            raise ValueError(f"Unknown UNet version: {version}")
    elif architecture == "deeplab":
        # build deeplab model using helper (smp)
        return build_deeplab(deeplab_variant, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


#######################################
### MODEL-SPECIFIC BUILDS & CLASSES ###


# --- MASK R-CNN (Instance Segmentation) ---
def build_maskrcnn(num_classes):
    # Load an instance segmentation model pre-trained on COCO + most recent weights
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Stop here if you are fine-tunning Faster-RCNN

    # Now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


# --- DEEPLABV3+ ---
# Supported backends:
#  - segmentation_models_pytorch (DeepLabV3Plus) if installed
#    https://smp.readthedocs.io/en/v0.1.3/_modules/segmentation_models_pytorch/deeplabv3/model.html
#  - more can be added later if needed (e.g. torchvision)
def build_deeplab(
    variant: str, num_classes: int = 2, encoder_name: str = None, encoder_weights: Optional[str] = "imagenet"
):
    """
    variant: 'smp'   -> segmentation_models_pytorch DeepLabV3Plus
    num_classes: number of output classes (2 in our case: foreground/background)
    encoder_name: name of encoder for SMP backend (e.g. 'resnet34' or 'resnet18')
    encoder_weights: weights for SMP encoder (None or 'imagenet')
    """
    variant = variant.lower()

    # DeepLabV3+ from SMP (segmentation_models_pytorch)
    if variant == "smp":
        # Choose a reasonable default encoder (e.g. 'resnet34') if not provided
        enc = "resnet34" if encoder_name is None else encoder_name
        print(f"DeepLabV3+ model from segmentation_models_pytorch with encoder '{enc}' and weights '{encoder_weights}'")
        model = smp.DeepLabV3Plus(encoder_name=enc, encoder_weights=encoder_weights, in_channels=3, classes=num_classes)
        return model
    # other backends can be added here (if needed later)


# --- U-NET VARIANTS ---
class UNetV1(nn.Module):
    """
    Original U-Net architecture -> Usage: UNetV1(in_channels=3, out_channels=2)
    """

    def __init__(self, in_channels=3, out_channels=2):
        super(UNetV1, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        self.enc1 = CBR(in_channels, 64)
        self.enc2 = CBR(64, 128)
        self.enc3 = CBR(128, 256)
        self.enc4 = CBR(256, 512)
        self.pool = nn.MaxPool2d(2)
        self.center = CBR(512, 1024)
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = CBR(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = CBR(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = CBR(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = CBR(128, 64)
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)  # 512x512
        e2 = self.enc2(self.pool(e1))  # 256x256
        e3 = self.enc3(self.pool(e2))  # 128x128
        e4 = self.enc4(self.pool(e3))  # 64x64
        center = self.center(self.pool(e4))  # 32x32
        d4 = self.dec4(torch.cat([self.up4(center), e4], 1))  # 64x64
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))  # 128x128
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))  # 256x256
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))  # 512x512
        out = self.final(d1)  # final output: (batch, out_channels, 512, 512)
        return out


class AttentionGate(nn.Module):
    """Simple attention gate used in Attention U-Net
    -> It takes the skip connection (x) and the gating signal (g) and produces
       an attention-weighted output of the same spatial size as x
    """

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm2d(1), nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # x: skip connection (batch, F_l, H, W)
        # g: gating signal (batch, F_g, H_g, W_g) -> will be up/down-sampled to x's size
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        # If necessary, upsample g1 to x1 spatial size
        if g1.shape[2:] != x1.shape[2:]:
            g1 = nn.functional.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        # multiply attention coefficients with the skip connection
        return x * psi


class UNetV2(nn.Module):
    """
    Attention U-Net -> Usage: UNetV2(in_channels=3, out_channels=2)
    """

    def __init__(self, in_channels=3, out_channels=2, base_filters=64):
        super(UNetV2, self).__init__()

        def CBR(in_ch, out_ch):
            return nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

        # Define number of filters at each level: 64, 128, 256, 512, 1024
        filters = [base_filters, base_filters * 2, base_filters * 4, base_filters * 8, base_filters * 16]

        # ENCODER LAYERS
        self.enc1 = CBR(in_channels, filters[0])
        self.enc2 = CBR(filters[0], filters[1])
        self.enc3 = CBR(filters[1], filters[2])
        self.enc4 = CBR(filters[2], filters[3])
        self.pool = nn.MaxPool2d(2)
        # CENTER LAYER
        self.center = CBR(filters[3], filters[4])
        # DECODER LAYERS WITH ATTENTION GATES
        # Decoder layer 4
        self.up4 = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)
        self.att4 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[3] // 2)
        self.dec4 = CBR(filters[4], filters[3])
        # Decoder layer 3
        self.up3 = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        self.att3 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[2] // 2)
        self.dec3 = CBR(filters[3], filters[2])
        # Decoder layer 2
        self.up2 = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        self.att2 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[1] // 2)
        self.dec2 = CBR(filters[2], filters[1])
        # Decoder layer 1
        self.up1 = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        self.att1 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.dec1 = CBR(filters[1], filters[0])
        # FINAL OUTPUT LAYER
        self.final = nn.Conv2d(filters[0], out_channels, 1)

    def forward(self, x):
        # Input x: (batch, 3, 512, 512) -> (B, C, H, W)
        e1 = self.enc1(x)  # 512x512
        e2 = self.enc2(self.pool(e1))  # 256x256
        e3 = self.enc3(self.pool(e2))  # 128x128
        e4 = self.enc4(self.pool(e3))  # 64x64
        center = self.center(self.pool(e4))  # 32x32

        u4 = self.up4(center)  # ConvTranspose to 64x64
        e4_att = self.att4(e4, u4)  # apply attention gate
        d4 = self.dec4(torch.cat([u4, e4_att], dim=1))  # CBR 64x64

        u3 = self.up3(d4)  # ConvTranspose to 128x128
        e3_att = self.att3(e3, u3)  # attention gate
        d3 = self.dec3(torch.cat([u3, e3_att], dim=1))  # CBR 128x128

        u2 = self.up2(d3)  # 256x256
        e2_att = self.att2(e2, u2)
        d2 = self.dec2(torch.cat([u2, e2_att], dim=1))

        u1 = self.up1(d2)  # 512x512
        e1_att = self.att1(e1, u1)
        d1 = self.dec1(torch.cat([u1, e1_att], dim=1))

        out = self.final(d1)  # final output: (batch, out_channels, 512, 512) -> (B, C, H, W)
        return out


class UNetV3(nn.Module):
    """UNetV3: Unet++ (Nested U-Net) using segmentation_models_pytorch
    Wraps `smp.UnetPlusPlus` and by default uses a pretrained ImageNet encoder (good for small datasets)
    Other encoder_name options:
    ["resnet18", "resnet34", "resnet50", "efficientnet-b0", "mobilenet_v2", ...]
    """

    def __init__(self, in_channels=3, out_channels=2, encoder_name="resnet34", encoder_weights="imagenet"):
        print(
            f"UNetV3 (Unet++) model from segmentation_models_pytorch with encoder '{encoder_name}' and weights '{encoder_weights}'"
        )
        super(UNetV3, self).__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )

    def forward(self, x):
        # smp models output logits shape (Batch, Classes, Height, Width)
        return self.model(x)
