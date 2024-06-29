import torch
from torch import nn, Tensor, LongTensor
from torch.nn import init
import torch.nn.functional as F
import torchvision
from einops import rearrange

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class get_contextual_information(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_k1 = nn.Conv2d(dim, dim, 1,  groups=dim)
        self.conv_k3 = nn.Conv2d(dim, dim, 3, stride=1, padding=1)
        self.conv_k5 = nn.Conv2d(dim, dim, 5, padding=4, groups=dim, dilation=2)
        self.conv_K7 = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)


        self.conv1 = nn.Conv2d(dim, dim // 4, 1)
        self.conv2 = nn.Conv2d(dim, dim // 4, 1)
        self.conv3 = nn.Conv2d(dim, dim // 4, 1)
        self.conv4 = nn.Conv2d(dim, dim // 4, 1)

        self.Channel_Integration = nn.Conv2d(2, 4, 7, padding=3)
        self.conv = nn.Conv2d(16, dim, 1)

    def forward(self, x):
        f1 = self.conv_k1(x)
        f2 = self.conv_k3(x)
        f3 = self.conv_k5(x)
        f4 = self.conv_K7(x)


        f1s = self.conv1(f1)
        f2s = self.conv2(f2)
        f3s = self.conv3(f3)
        f4s = self.conv4(f4)


        contextual_information = torch.cat([f1s,f2s,f3s,f4s], dim=1)

        avg = torch.mean(contextual_information, dim=1, keepdim=True)

        max, _ = torch.max(contextual_information, dim=1, keepdim=True)

        atten = torch.cat([avg, max], dim=1)

        sig = self.Channel_Integration(atten).sigmoid()

        attn = f1 * sig[:, 0, :, :].unsqueeze(1) + f2 * sig[:, 1, :, :].unsqueeze(1)+ f3 * sig[:, 2, :, :].unsqueeze(1)+ f4 * sig[:, 3, :, :].unsqueeze(1)

        attn = self.conv(attn)

        return x * attn



class CISB(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.conv1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.get_contextual_information = get_contextual_information(d_model)
        self.conv2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):

        x = self.conv1(x)

        x = self.activation(x)

        x = self.get_contextual_information(x)
        
        x = self.conv2(x)

        return x



class RFAConv_CISB(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.get_weight = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, padding=kernel_size // 2, stride=stride),
                                        nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=1,
                                                  groups=in_channel, bias=False))
        self.generate_feature = nn.Sequential(
            nn.Conv2d(in_channel, in_channel * (kernel_size ** 2), kernel_size=kernel_size, padding=kernel_size // 2,
                      stride=stride, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel * (kernel_size ** 2)),
            nn.ReLU())

        self.conv = Conv(in_channel, out_channel, k=kernel_size, s=kernel_size, p=0)
        self.lsk = CISB(in_channel)



    def forward(self, x):


        x = self.lsk(x)

        b, c = x.shape[0:2]

        weight = self.get_weight(x)

        h, w = weight.shape[2:]

        weighted = weight.view(b, c, self.kernel_size ** 2, h, w).softmax(2)  # b c*kernel**2,h,w ->  b c k**2 h w

        feature = self.generate_feature(x).view(b, c, self.kernel_size ** 2, h, w)  # b c*kernel**2,h,w ->  b c k**2 h w

        weighted_data = feature * weighted

        conv_data = rearrange(weighted_data, 'b c (n1 n2) h w -> b c (h n1) (w n2)', n1=self.kernel_size,
                              # b c k**2 h w ->  b c h*k w*k
                              n2=self.kernel_size)

        return self.conv(conv_data)