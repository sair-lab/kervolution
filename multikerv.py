# -* coding=utf-8 *-
# Kervolution Neural Networks
# This file is part of the project of Kervolutional Neural Networks
# It implements the kervolution for 4D tensors in Pytorch
# Copyright (C) 2018 [Wang, Chen] <wang.chen@zoho.com>
# Nanyang Technological University (NTU), Singapore.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import torch
import torch.nn as nn
import sys
sys.path.append(".")
if __name__ == '__main__':
    import kervolution
else:
    from . import kervolution
from torch.autograd import Variable
from torch.nn.modules.module import Module


class MultiKerv2d(nn.Module):
    '''
    multiple kervolution on multiple output channels:
    '''
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            mapping=['translation'], 
            kernel_type=['linear'],
            learnable_kernel=[False], 
            kernel_regularizer=[False],
            alpha=0.03, balance=2, power=3, sigma=2, gamma=1,
            kernel_out_channels=[-1]):
        super(MultiKerv2d, self).__init__()

        assert(out_channels==sum(kernel_out_channels))
        assert(len(mapping)==len(kernel_type)==len(kernel_out_channels))
        assert(len(mapping)==len(learnable_kernel)==len(kernel_regularizer))

        self.kerv2d = []
        self.output = [None]*len(mapping)

        for i, channels in enumerate(kernel_out_channels):
            assert(channels>0)
            self.kerv2d.append(nn.Sequential(
                nn.Kerv2d(in_channels, channels, kernel_size, stride, padding, dilation, groups, bias,
                    mapping = mapping[i],
                    kernel_type = kernel_type[i],
                    learnable_kernel = learnable_kernel[i],
                    kernel_regularizer = kernel_regularizer[i],
                    alpha=alpha, balance=balance, power=power, sigma=sigma, gamma=gamma),
                nn.BatchNorm2d(channels)
                )
            )
            self.kerv2d[i].cuda()

    def forward(self, input):
        for i, kerv in enumerate(self.kerv2d):
            self.output[i] = kerv(input)
        return torch.cat(self.output, 1)

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU."""
        for kerv in self.kerv2d:
            kerv.cuda()
        return self._apply(lambda t: t.cuda(device_id))

    def cpu(self):
        """Moves all model parameters and buffers to the CPU."""
        for kerv in self.kerv2d:
            kerv.cpu()
        return self._apply(lambda t: t.cpu())


nn.MultiKerv2d = MultiKerv2d


if __name__ == '__main__':
    kerv = nn.MultiKerv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1, padding=1,
                            mapping=['translation','translation'],
                            kernel_type=['linear','polynomial'],
                            kernel_out_channels=[2, 2],
                            learnable_kernel=[False, False], 
                            kernel_regularizer=[False, False],
                            alpha=0.03, balance=2, power=3, sigma=2, gamma=1).cuda()

    input=Variable(torch.randn(50,2,30,30)).cuda()
    output=kerv(input)
    print(output.size())

    conv= nn.Conv2d(2,4,3,1,1)
    print(conv.weight.size())