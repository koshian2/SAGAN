import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm

def init_xavier_uniform(layer):
    if hasattr(layer, "weight"):
        torch.nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, "bias"):
        if hasattr(layer.bias, "data"):       
            layer.bias.data.fill_(0)

class ConditionalBatchNorm2d(nn.Module):
  def __init__(self, num_features, num_classes, eps=1e-4, momentum=0.1):
    super().__init__()
    self.num_features = num_features
    self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps, momentum=momentum)
    self.gamma_embed = nn.Linear(num_classes, num_features, bias=False)
    self.beta_embed = nn.Linear(num_classes, num_features, bias=False)

    self.gamma_embed.apply(init_xavier_uniform)
    self.beta_embed.apply(init_xavier_uniform)

  def forward(self, x, y):
    out = self.bn(x)
    gamma = self.gamma_embed(y) + 1
    beta = self.beta_embed(y)
    out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
    return out

## Spectral Conv layer
class SNConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.layer = spectral_norm(nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding))
        self.layer.apply(init_xavier_uniform)

    def forward(self, inputs):
        return self.layer(inputs)

class SNLinear(nn.Module):
    def __init__(self, in_dims, out_dims, bias=True):
        super().__init__()
        self.layer = spectral_norm(nn.Linear(in_dims, out_dims, bias=bias))
        self.layer.apply(init_xavier_uniform)

    def forward(self, inputs):
        return self.layer(inputs)

## Self attention
class SelfAttention(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv_theta = SNConv2d(dims, dims // 8, kernel_size=1)
        self.conv_phi = SNConv2d(dims, dims // 8, kernel_size=1)
        self.conv_g = SNConv2d(dims, dims // 2, kernel_size=1)
        self.conv_attn = SNConv2d(dims // 2, dims, kernel_size=1)
        self.sigma_ratio = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, inputs):
        batch, ch, height, width = inputs.size()
        # theta path
        theta = self.conv_theta(inputs)
        theta = theta.view(batch, ch // 8, height * width).permute([0, 2, 1])  # (B, HW, C/8)        
        # phi path
        phi = self.conv_phi(inputs)
        phi = F.max_pool2d(phi, kernel_size=2)  # (B, C/8, H/2, W/2)
        phi = phi.view(batch, ch // 8, height * width // 4)  # (B, C/8, HW/4)
        # attention
        attn = torch.bmm(theta, phi)  # (B, HW, HW/4)
        attn = F.softmax(attn, dim=-1)
        # g path
        g = self.conv_g(inputs)
        g = F.max_pool2d(g, kernel_size=2)  # (B, C/2, H/2, W/2)
        g = g.view(batch, ch // 2, height * width // 4).permute([0, 2, 1])  # (B, HW/4, C/2)

        attn_g = torch.bmm(attn, g)  # (B, HW, C/2)
        attn_g = attn_g.permute([0, 2, 1]).view(batch, ch // 2, height, width)  # (B, C/2, H, W)
        attn_g = self.conv_attn(attn_g)
        return inputs + self.sigma_ratio * attn_g

## Generator block for SAGAN (based on SNGAN)
class GeneratorSNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, upsampling, n_classes=0):
        super().__init__()
        self.conv1 = SNConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = SNConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.upsampling = upsampling
        if n_classes == 0:
            self.bn1 = nn.BatchNorm2d(in_ch)
            self.bn2 = nn.BatchNorm2d(out_ch)
        else:
            self.bn1 = ConditionalBatchNorm2d(in_ch, n_classes)
            self.bn2 = ConditionalBatchNorm2d(out_ch, n_classes)
        if in_ch != out_ch or upsampling > 1:
            self.shortcut_conv = SNConv2d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut_conv = None

    def forward(self, inputs, label_onehots=None):
        # main
        if label_onehots is not None:
            x = self.bn1(inputs, label_onehots)
        else:
            x = self.bn1(inputs)
        x = F.relu(x)

        if self.upsampling > 1:
            x = F.interpolate(x, scale_factor=self.upsampling)
        x = self.conv1(x)

        if label_onehots is not None:
            x = self.bn2(x, label_onehots)
        else:
            x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2(x)

        # short cut
        if self.upsampling > 1:
            shortcut = F.interpolate(inputs, scale_factor=self.upsampling)
        else:
            shortcut = inputs
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)
        # residual add
        return x + shortcut
        
## Discriminator Block
class SNEmbedding(nn.Module):
    def __init__(self, n_classes, out_dims):
        super().__init__()
        self.linear = SNLinear(n_classes, out_dims, bias=False)

    def forward(self, base_features, output_logits, label_onehots):
        wy = self.linear(label_onehots)
        weighted = torch.sum(base_features * wy, dim=1, keepdim=True)
        return output_logits + weighted

class DiscriminatorSNResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, downsampling):
        super().__init__()
        self.conv1 = SNConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = SNConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.downsampling = downsampling
        if in_ch != out_ch or downsampling > 1:
            self.shortcut_conv = SNConv2d(in_ch, out_ch, kernel_size=1, padding=0)
        else:
            self.shortcut_conv = None

    def forward(self, inputs):
        x = F.relu(inputs)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        # short cut
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(inputs)
        else:
            shortcut = inputs
        if self.downsampling > 1:
            x = F.avg_pool2d(x, kernel_size=self.downsampling)
            shortcut = F.avg_pool2d(shortcut, kernel_size=self.downsampling)
        # residual add
        return x + shortcut

class DiscriminatorOptimizedBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = SNConv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = SNConv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.shortcut_conv = SNConv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = F.avg_pool2d(self.conv2(x), kernel_size=2)
        # shortcut
        shorcut = self.shortcut_conv(F.avg_pool2d(inputs, kernel_size=2))
        return x + shorcut
