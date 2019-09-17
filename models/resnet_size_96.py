import torch
from torch import nn
import torch.nn.functional as F
from models.basic_module import GeneratorSNResidualBlock, DiscriminatorSNResidualBlock, DiscriminatorOptimizedBlock
from models.basic_module import SNEmbedding, SelfAttention, ConditionalBatchNorm2d, SNLinear, SNConv2d

class Generator(nn.Module):
    def __init__(self, n_classes_g=0):
        super().__init__()
        self.dense = SNLinear(128, 3 * 3 * 1024)
        self.block1 = GeneratorSNResidualBlock(1024, 1024, 2, n_classes=n_classes_g) # 6x6
        self.block2 = GeneratorSNResidualBlock(1024, 512, 2, n_classes=n_classes_g) # 12x12
        self.block3 = GeneratorSNResidualBlock(512, 256, 2, n_classes=n_classes_g)  # 24x24
        self.attn = SelfAttention(256) # feat 24
        self.block4 = GeneratorSNResidualBlock(256, 128, 2, n_classes=n_classes_g)
        self.block5 = GeneratorSNResidualBlock(128, 64, 2, n_classes=n_classes_g)
        self.bn_out = ConditionalBatchNorm2d(64, n_classes_g) if n_classes_g > 0 else nn.BatchNorm2d(64)
        self.out = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, inputs, y=None):
        x = self.dense(inputs).view(inputs.size(0), 1024, 3, 3)
        x = self.block3(self.block2(self.block1(x, y), y), y)
        x = self.attn(x)
        x = self.block5(self.block4(x, y), y)
        x = self.bn_out(x, y) if y is not None else self.bn_out(x)
        return self.out(x)

class Discriminator(nn.Module):
    def __init__(self, n_classes_d=0):
        super().__init__()
        self.optim_block = DiscriminatorOptimizedBlock(3, 64) # 48x48
        self.block1 = DiscriminatorSNResidualBlock(64, 128, 2)  # 24x24
        self.attn = SelfAttention(128) # feat 24
        self.block2 = DiscriminatorSNResidualBlock(128, 256, 2) # 12x12
        self.block3 = DiscriminatorSNResidualBlock(256, 512, 2) # 6x6
        self.block4 = DiscriminatorSNResidualBlock(512, 1024, 2) # 3x3
        self.block5 = DiscriminatorSNResidualBlock(1024, 1024, 1)
        self.dense = nn.Linear(1024, 1)
        if n_classes_d > 0:
            self.sn_embedding = SNEmbedding(n_classes_d, 1024)
        else:
            self.sn_embedding = None

    def forward(self, inputs, y=None):
        x = self.attn(self.block1(self.optim_block(inputs)))
        x = self.block5(self.block4(self.block3(self.block2(x))))
        x = F.relu(x)
        features = torch.sum(x, dim=(2,3)) # gloobal sum pooling
        x = self.dense(features)
        if self.sn_embedding is not None:
            x = self.sn_embedding(features, x, y)
        return x

