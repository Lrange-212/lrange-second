import torch
import torch.nn as nn
from torch import Tensor


class PConv(nn.Module):
    """
    部分卷积层。

    参数：
        dim (int): 输入/输出通道数。
        n_div (int): 部分比例的倒数。
        forward (str): 前向传播方式，可以是'split_cat'或'slicing'。
        kernel_size (int): 卷积核大小。
    """

    def __init__(self,
                 dim: int,
                 n_div: int,
                 forward: str = "split_cat",
                 kernel_size: int = 3) -> None:
        super().__init__()

        if dim % n_div != 0:
            raise ValueError("dim必须能被n_div整除。")

        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv

        self.conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False)

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError("无效的前向传播类型，必须是'split_cat'或'slicing'。")

    def forward_slicing(self, x: Tensor) -> Tensor:
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), dim=1)
        return x
