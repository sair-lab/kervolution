from .kervolution import Kerv2d
from .kervolution import Kerv1d
from .multikerv import MultiKerv2d
from .tools import Timer
import torch.nn as nn
nn.Kerv2d = Kerv2d
nn.Kerv1d = Kerv1d