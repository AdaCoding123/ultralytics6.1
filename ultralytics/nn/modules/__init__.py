# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3,CBRM, Shuffle_Block,ConvNeXt_Stem, ConvNeXt_Block, ConvNeXt_Downsample,
                    SGBlock,Conv_BN_HSwish, MobileNetV3_InvertedResidual,MV2Block, MobileViTBlock, SimAM, ECA, SpatialGroupEnhance, 
                    TripletAttention, CoordAtt, GAMAttention,SE, ShuffleAttention, SKAttention, DoubleAttention, CoTAttention, EffectiveSEModule,
                    GlobalContext, GatherExcite, MHSA, S2Attention, NAMAttention, CrissCrossAttention,SequentialPolarizedSelfAttention, 
                    ParallelPolarizedSelfAttention, ParNetAttention,BiFPN_Add3,BiFPN_Add2,BoTNet,PatchMerging, PatchEmbed, SwinStage)

from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment
from .addModules.transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP','CBRM', 
           'Shuffle_Block','ConvNeXt_Stem', 'ConvNeXt_Block', 'ConvNeXt_Downsample','SGBlock','Conv_BN_HSwish', 'MobileNetV3_InvertedResidual',
           'MV2Block', 'MobileViTBlock', 'SimAM', 'ECA', 'SpatialGroupEnhance', 'TripletAttention', 'CoordAtt', 'GAMAttention',
           'SE', 'ShuffleAttention', 'SKAttention', 'DoubleAttention', 'CoTAttention', 'EffectiveSEModule',
           'GlobalContext', 'GatherExcite', 'MHSA', 'S2Attention', 'NAMAttention', 'CrissCrossAttention',
           'SequentialPolarizedSelfAttention', 'ParallelPolarizedSelfAttention', 'ParNetAttention','BiFPN_Add3','BiFPN_Add2',
           'PatchMerging', 'PatchEmbed', 'SwinStage','BoTNet')