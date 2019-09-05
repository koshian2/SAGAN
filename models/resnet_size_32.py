import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module import GeneratorSNResidualBlock, DiscriminatorSNResidualBlock, DiscriminatorOptimizedBlock
from models.basic_module import SNEmbedding, SelfAttention, ConditionalBatchNorm2d, SNLinear, SNConv2d

class Generator(nn.Module):
    def __init__(self, enable_conditional=False, use_self_attention=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.dense = SNLinear(128, 4 * 4 * 256) # 4x4
        self.block1 = GeneratorSNResidualBlock(256, 256, 2, n_classes=n_classes) # 8x8
        self.atten = SelfAttention(256) if use_self_attention else None # feat 8 
        self.block2 = GeneratorSNResidualBlock(256, 256, 2, n_classes=n_classes) # 16x16
        self.block3 = GeneratorSNResidualBlock(256, 256, 2, n_classes=n_classes) # 32x32
        self.bn_out = ConditionalBatchNorm2d(256, n_classes) if enable_conditional else nn.BatchNorm2d(256)
        self.out_conv = nn.Sequential(
            nn.ReLU(True),
            SNConv2d(256, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 256, 4, 4)
        x = self.block1(x, y)
        if self.atten is not None: x = self.atten(x) 
        x = self.block3(self.block2(x, y), y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)
        return self.out_conv(x)

class Discriminator(nn.Module):
    def __init__(self, enable_conditional=False, use_self_attention=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.optim_block = DiscriminatorOptimizedBlock(3, 128) # 16x16
        self.block1 = DiscriminatorSNResidualBlock(128, 128, 2)  # 8x8
        self.atten = SelfAttention(128) if use_self_attention else None # 8x8
        self.block2 = DiscriminatorSNResidualBlock(128, 128, 1)
        self.block3 = DiscriminatorSNResidualBlock(128, 128, 1)
        self.dense = SNLinear(128, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, 128)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.block1(self.optim_block(inputs))
        if self.atten is not None: x = self.atten(x)
        x = self.block3(self.block2(x))
        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # global sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x
