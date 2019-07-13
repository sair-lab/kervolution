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

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.nn.functional import conv2d
import torch.nn.functional as F
import numpy as np


class Kerv2d(nn.Conv2d):
    '''
    kervolution with following options:
    kernel_type: [linear, polynomial, gaussian, etc.]
    default is convolution:
             kernel_type --> linear,
    balance, power, gamma is valid only when the kernel_type is specified
    if learnable_kernel = True,  they just be the initial value of learable parameters
    if learnable_kernel = False, they are the value of kernel_type's parameter
    the parameter [power] cannot be learned due to integer limitation
    '''
    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True,
            kernel_type='linear', learnable_kernel=False, kernel_regularizer=False,
            balance=1, power=3, gamma=1):

        super(Kerv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.kernel_type = kernel_type
        self.learnable_kernel, self.kernel_regularizer = learnable_kernel, kernel_regularizer
        self.balance, self.power, self.gamma = balance, power, gamma

        # parameter for kernel type
        if learnable_kernel == True:
            self.balance = nn.Parameter(torch.cuda.FloatTensor([balance] * out_channels), requires_grad=True).view(-1, 1)
            self.gamma   = nn.Parameter(torch.cuda.FloatTensor([gamma]   * out_channels), requires_grad=True).view(-1, 1)

    def forward(self, input):

        minibatch, in_channels, input_width, input_hight = input.size()
        assert(in_channels == self.in_channels)
        input_unfold = F.unfold(input, kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        input_unfold = input_unfold.view(minibatch, 1, self.kernel_size[0]*self.kernel_size[1]*self.in_channels, -1)
        weight_flat  = self.weight.view(self.out_channels, -1, 1)
        output_width = (input_width - self.kernel_size[0] + 2 * self.padding[0]) // self.stride[0] + 1
        output_hight = (input_hight - self.kernel_size[1] + 2 * self.padding[1]) // self.stride[1] + 1

        if self.kernel_type == 'linear':
            output = (input_unfold * weight_flat).sum(dim=2)

        elif self.kernel_type == 'manhattan':
            output = -((input_unfold - weight_flat).abs().sum(dim=2))

        elif self.kernel_type == 'euclidean':
            output = -(((input_unfold - weight_flat)**2).sum(dim=2))

        elif self.kernel_type == 'polynomial':
            output = ((input_unfold * weight_flat).sum(dim=2) + self.balance)**self.power

        elif self.kernel_type == 'gaussian':
            output = (-self.gamma*((input_unfold - weight_flat)**2).sum(dim=2)).exp() + 0

        else:
            raise NotImplementedError(self.kernel_type+' kervolution not implemented')

        if self.bias is not None:
            output += self.bias.view(self.out_channels, -1)

        return output.view(minibatch, self.out_channels, output_width, output_hight)


class Kerv1d(nn.Conv1d):
    r"""Applies a 1D kervolution over an input signal composed of several input
        planes.

        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 0
            padding_mode (string, optional). Accepted values `zeros` and `circular` Default: `zeros`
            dilation (int or tuple, optional): Spacing between kernel
                elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
            kernel_type (str), Default: 'linear'
            learnable_kernel (bool): Learnable kernel parameters.  Default: False 
            balance=1, power=3, gamma=1

        Shape:
            - Input: :math:`(N, C_{in}, L_{in})`
            - Output: :math:`(N, C_{out}, L_{out})` where

            .. math::
                L_{out} = \left\lfloor\frac{L_{in} + 2 \times \text{padding} - \text{dilation}
                            \times (\text{kernel\_size} - 1) - 1}{\text{stride}} + 1\right\rfloor

        Examples::

            >>> m = Kerv1d(16, 33, 3, kernel_type='polynomial', learnable_kernel=True)
            >>> input = torch.randn(20, 16, 50)
            >>> output = m(input)

        .. _kervolution:
            https://arxiv.org/pdf/1904.03955.pdf
        """

    def __init__(self, in_channels, out_channels, kernel_size, 
            stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
            kernel_type='linear', learnable_kernel=False, balance=1, power=3, gamma=1):

        super(Kerv1d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.kernel_type, self.learnable_kernel = kernel_type, learnable_kernel
        self.balance, self.power, self.gamma = balance, power, gamma
        self.unfold = nn.Unfold((kernel_size,1), (dilation,1), (padding, 0), (stride,1))

        # parameter for kernels
        # if learnable_kernel == True:
        # self.balance = nn.Parameter(torch.FloatTensor([balance] * out_channels)).view(-1, 1)
        # self.gamma   = nn.Parameter(torch.FloatTensor([gamma]   * out_channels)).view(-1, 1)


    def forward(self, input):
        input = self.unfold(input.unsqueeze(-1)).unsqueeze(1)
        weight  = self.weight.view(self.out_channels, -1, 1)

        if self.kernel_type == 'linear':
            output = (input * weight).sum(dim=2)

        elif self.kernel_type == 'manhattan':
            output = -((input - weight).abs().sum(dim=2))

        elif self.kernel_type == 'euclidean':
            output = -(((input - weight)**2).sum(dim=2))

        elif self.kernel_type == 'polynomial':
            output = ((input * weight).sum(dim=2) + self.balance)**self.power

        elif self.kernel_type == 'gaussian':
            output = (-self.gamma*((input - weight)**2).sum(dim=2)).exp() + 0

        else:
            raise NotImplementedError(self.kernel_type+' Kerv1d not implemented')

        if self.bias is not None:
            output += self.bias.view(self.out_channels, -1)

        return output


    def cuda(self, device=None):
        if self.learnable_kernel == True:
            self.balance = self.balance.cuda(device)
            self.gamma = self.gamma.cuda(device)
        return self._apply(lambda t: t.cuda(device))


if __name__ == '__main__':
    kerv = Kerv2d(in_channels=2,              # input height
                  out_channels=3,             # n_filters
                  kernel_size=3,              # filter size
                  stride=1,                   # filter movement/step
                  padding=1,                  # input padding
                  kernel_type='polynomial',   # kernel type
                  learnable_kernel=True)      # enable learning parameters

    n_batch, in_channels, n_feature = 5, 2, 5
    x = torch.FloatTensor(n_batch, in_channels, n_feature).random_().cuda()
    kerv1d = Kerv1d(in_channels=in_channels, out_channels=2, kernel_size=3, kernel_type='polynomial', learnable_kernel=True).cuda()
    y = kerv1d(x)
    print(x.shape, y.shape)
