import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import ConvertImageDtype
import torchvision.models.detection as detection
from implicit_layers import *

def model_imp(feature_alignment=True, anchor_refinement=True, prediction_refinement=True):
    #Create the baseline fasterrcnn model
    model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=True)
    state_dict = torch.load('seed_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)

    #Add implicit knowledge as indicated by parameters
    # Feature Alignment
    if feature_alignment:
        for i in range(len(model.backbone.fpn.inner_blocks)):
            # model.backbone.fpn.inner_blocks[i] = ImplicitWrapperMul(model.backbone.fpn.inner_blocks[i], (1, 256, 1, 1))
            model.backbone.fpn.inner_blocks[i] = ImplicitWrapperAdd(model.backbone.fpn.inner_blocks[i], (1, 256, 1, 1))
    
    # Anchor Refinement
    if anchor_refinement:
        # model.rpn.head.bbox_pred = ImplicitWrapperMul(model.rpn.head.bbox_pred, (1, 60, 1, 1))
        model.rpn.head.bbox_pred = ImplicitWrapperAdd(model.rpn.head.bbox_pred, (1, 60, 1, 1))



    # Prediction Refinement
    if prediction_refinement:
        # model.roi_heads.box_predictor.cls_score = ImplicitWrapperMul(model.roi_heads.box_predictor.cls_score, (1, 91))
        model.roi_heads.box_predictor.cls_score = ImplicitWrapperAdd(model.roi_heads.box_predictor.cls_score, (1, 91))
        # model.roi_heads.box_predictor.bbox_pred = ImplicitWrapperMul(model.roi_heads.box_predictor.bbox_pred, (1, 364))
        model.roi_heads.box_predictor.bbox_pred = ImplicitWrapperAdd(model.roi_heads.box_predictor.bbox_pred, (1, 364))
    
    return model
""" 
FasterRCNN(
  (transform): GeneralizedRCNNTransform(
      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      Resize(min_size=(800,), max_size=1333, mode='bilinear')
  )
  (backbone): BackboneWithFPN(
    (body): IntermediateLayerGetter(
      (0): ConvNormActivation(
        (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        (1): FrozenBatchNorm2d(16, eps=1e-05)
        (2): Hardswish()
      )
      (1): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=16, bias=False)
            (1): FrozenBatchNorm2d(16, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(16, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(16, eps=1e-05)
          )
        )
      )
      (2): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(16, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(64, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=64, bias=False)
            (1): FrozenBatchNorm2d(64, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (2): ConvNormActivation(
            (0): Conv2d(64, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(24, eps=1e-05)
          )
        )
      )
      (3): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(72, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(72, 72, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=72, bias=False)
            (1): FrozenBatchNorm2d(72, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (2): ConvNormActivation(
            (0): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(24, eps=1e-05)
          )
        )
      )
      (4): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(72, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(72, 72, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=72, bias=False)
            (1): FrozenBatchNorm2d(72, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(72, 24, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(24, 72, kernel_size=(1, 1), stride=(1, 1))
            (activation): ReLU()
            (scale_activation): Hardsigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(72, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(40, eps=1e-05)
          )
        )
      )
      (5): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(120, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
            (1): FrozenBatchNorm2d(120, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
            (activation): ReLU()
            (scale_activation): Hardsigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(40, eps=1e-05)
          )
        )
      )
      (6): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(40, 120, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(120, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (1): ConvNormActivation(
            (0): Conv2d(120, 120, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=120, bias=False)
            (1): FrozenBatchNorm2d(120, eps=1e-05)
            (2): ReLU(inplace=True)
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(120, 32, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(32, 120, kernel_size=(1, 1), stride=(1, 1))
            (activation): ReLU()
            (scale_activation): Hardsigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(120, 40, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(40, eps=1e-05)
          )
        )
      )
      (7): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(40, 240, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(240, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(240, 240, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=240, bias=False)
            (1): FrozenBatchNorm2d(240, eps=1e-05)
            (2): Hardswish()
          )
          (2): ConvNormActivation(
            (0): Conv2d(240, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(80, eps=1e-05)
          )
        )
      )
      (8): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(80, 200, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(200, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(200, 200, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=200, bias=False)
            (1): FrozenBatchNorm2d(200, eps=1e-05)
            (2): Hardswish()
          )
          (2): ConvNormActivation(
            (0): Conv2d(200, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(80, eps=1e-05)
          )
        )
      )
      (9): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(184, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
            (1): FrozenBatchNorm2d(184, eps=1e-05)
            (2): Hardswish()
          )
          (2): ConvNormActivation(
            (0): Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(80, eps=1e-05)
          )
        )
      )
      (10): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(80, 184, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(184, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(184, 184, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=184, bias=False)
            (1): FrozenBatchNorm2d(184, eps=1e-05)
            (2): Hardswish()
          )
          (2): ConvNormActivation(
            (0): Conv2d(184, 80, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(80, eps=1e-05)
          )
        )
      )
      (11): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(80, 480, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(480, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(480, 480, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=480, bias=False)
            (1): FrozenBatchNorm2d(480, eps=1e-05)
            (2): Hardswish()
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(480, 120, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(120, 480, kernel_size=(1, 1), stride=(1, 1))
            (activation): ReLU()
            (scale_activation): Hardsigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(480, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(112, eps=1e-05)
          )
        )
      )
      (12): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(672, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(672, 672, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=672, bias=False)
            (1): FrozenBatchNorm2d(672, eps=1e-05)
            (2): Hardswish()
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
            (activation): ReLU()
            (scale_activation): Hardsigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(672, 112, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(112, eps=1e-05)
          )
        )
      )
      (13): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(112, 672, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(672, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(672, 672, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), groups=672, bias=False)
            (1): FrozenBatchNorm2d(672, eps=1e-05)
            (2): Hardswish()
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(672, 168, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(168, 672, kernel_size=(1, 1), stride=(1, 1))
            (activation): ReLU()
            (scale_activation): Hardsigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(672, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(160, eps=1e-05)
          )
        )
      )
      (14): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(960, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
            (1): FrozenBatchNorm2d(960, eps=1e-05)
            (2): Hardswish()
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
            (activation): ReLU()
            (scale_activation): Hardsigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(160, eps=1e-05)
          )
        )
      )
      (15): InvertedResidual(
        (block): Sequential(
          (0): ConvNormActivation(
            (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(960, eps=1e-05)
            (2): Hardswish()
          )
          (1): ConvNormActivation(
            (0): Conv2d(960, 960, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2), groups=960, bias=False)
            (1): FrozenBatchNorm2d(960, eps=1e-05)
            (2): Hardswish()
          )
          (2): SqueezeExcitation(
            (avgpool): AdaptiveAvgPool2d(output_size=1)
            (fc1): Conv2d(960, 240, kernel_size=(1, 1), stride=(1, 1))
            (fc2): Conv2d(240, 960, kernel_size=(1, 1), stride=(1, 1))
            (activation): ReLU()
            (scale_activation): Hardsigmoid()
          )
          (3): ConvNormActivation(
            (0): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): FrozenBatchNorm2d(160, eps=1e-05)
          )
        )
      )
      (16): ConvNormActivation(
        (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        (1): FrozenBatchNorm2d(960, eps=1e-05)
        (2): Hardswish()
      )
    )

    
    (fpn): FeaturePyramidNetwork(
      (inner_blocks): ModuleList(
        (0): Conv2d(160, 256, kernel_size=(1, 1), stride=(1, 1))
        (1): Conv2d(960, 256, kernel_size=(1, 1), stride=(1, 1))
      )
      (layer_blocks): ModuleList(
        (0): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      )
      (extra_blocks): LastLevelMaxPool()
    )
  )
  (rpn): RegionProposalNetwork(
    (anchor_generator): AnchorGenerator()
    (head): RPNHead(
      (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (cls_logits): Conv2d(256, 15, kernel_size=(1, 1), stride=(1, 1))
      (bbox_pred): Conv2d(256, 60, kernel_size=(1, 1), stride=(1, 1))
    )
  )
  (roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=(7, 7), sampling_ratio=2)
    (box_head): TwoMLPHead(
      (fc6): Linear(in_features=12544, out_features=1024, bias=True)
      (fc7): Linear(in_features=1024, out_features=1024, bias=True)
    )
    (box_predictor): FastRCNNPredictor(
      (cls_score): Linear(in_features=1024, out_features=91, bias=True)
      (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
    )
  )
) """
