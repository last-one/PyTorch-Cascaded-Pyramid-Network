import base_model
import layer_utils
import torch.nn as nn
import torch
from global_refine_net import *

class CPN(nn.Module):

    def __init__(self, num_point):
        super(CPN, self).__init__()
        
        self.resnet101 = base_model.resnet101(pretrained=True)

        self.global_net = globalNet([2048, 1024, 512, 256], (64, 64), num_point)

        self.refine_net = refineNet(256, (64, 64), num_point)

    def forward(self, x):

        res_out = self.resnet101(x)

        global_re, global_out = self.global_net(res_out)

        refine_out = self.refine_net(global_re)

        return global_out, refine_out
