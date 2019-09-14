import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module import GeneratorSNResidualBlock, DiscriminatorSNResidualBlock, DiscriminatorOptimizedBlock
from models.basic_module import SNEmbedding, SelfAttention, ConditionalBatchNorm2d, SNLinear, SNConv2d

class Generator(nn.Module):
    def __init__(self, enable_conditional=False, use_self_attention=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.dense = SNLinear(128, 6 * 6 * 512)
        self.block1 = GeneratorSNResidualBlock(512, 256, 2, n_classes=n_classes)
        self.atten = SelfAttention(256) if use_self_attention else None # feat 12
        self.block2 = GeneratorSNResidualBlock(256, 128, 2, n_classes=n_classes)
        self.block3 = GeneratorSNResidualBlock(128, 64, 2, n_classes=n_classes)
        self.bn_out = ConditionalBatchNorm2d(64, n_classes) if enable_conditional else nn.BatchNorm2d(64)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 512, 6, 6)
        x = self.block1(x, y)
        if self.atten is not None: x = self.atten(x) 
        x = self.block3(self.block2(x, y), y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, enable_conditional=False, use_self_attention=False):
        super().__init__()
        n_classes = 10 if enable_conditional else 0
        self.optim_block = DiscriminatorOptimizedBlock(3, 64) # 24x24
        self.block1 = DiscriminatorSNResidualBlock(64, 128, 2) # 12x12
        self.atten = SelfAttention(128) if use_self_attention else None # 12x12        
        self.block2 = DiscriminatorSNResidualBlock(128, 256, 2)  # 6x6
        self.block3 = DiscriminatorSNResidualBlock(256, 512, 2) # 3x3
        self.block4 = DiscriminatorSNResidualBlock(512, 1024, 1)
        self.dense = SNLinear(1024, 1)
        if n_classes > 0:
            self.sn_embedding = SNEmbedding(n_classes, 1024)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.block1(self.optim_block(inputs))
        if self.atten is not None: x = self.atten(x)
        x = self.block4(self.block3(self.block2(x)))
        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x
    