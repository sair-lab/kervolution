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
    mapping: [translation, polar, logpolar, random]
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
            # return conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            output = (input_unfold * weight_flat).sum(dim=2)

        elif self.kernel_type == 'manhattan':
            output = -((input_unfold - weight_flat).abs().sum(dim=2))

        elif self.kernel_type == 'euclidean':
            output = -(((input_unfold - weight_flat)**2).sum(dim=2))

        elif self.kernel_type == 'polynomial':
            output = ((input_unfold * weight_flat).sum(dim=2) + self.balance)**self.power

        elif self.kernel_type == 'gaussian':
            output = (-self.gamma*((input_unfold - weight_flat).pow(2).sum(dim=2))).exp()
        else:
            raise NotImplementedError(self.kernel_type+' kervolution not implemented')

        if self.bias is not None:
            output += self.bias.view(self.out_channels, -1)

        return output.view(minibatch, self.out_channels, output_width, output_hight)


nn.Kerv2d = Kerv2d

if __name__ == '__main__':
    kerv = nn.Kerv2d(in_channels=2,              # input height
                     out_channels=3,             # n_filters
                     kernel_size=3,              # filter size
                     stride=1,                   # filter movement/step
                     padding=1,                  # input padding
                     kernel_type='polynomial',   # kernel type
                     learnable_kernel=True)      # enable learning parameters
