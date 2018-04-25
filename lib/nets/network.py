import base_model
import layer_utils
import torch.nn as nn
import torch
from global_refine_net import *

class CPN(nn.Module):

    def __init__(self, output_shape, num_points, pretrained=True):
        super(CPN, self).__init__()
        
        self.resnet101 = base_model.resnet101(pretrained=pretrained)

        self.global_net = globalNet([2048, 1024, 512, 256], output_shape, num_points)

        self.refine_net = refineNet(256, output_shape, num_points)

    def forward(self, x):

        res_out = self.resnet101(x)

        global_re, global_out = self.global_net(res_out)

        refine_out = self.refine_net(global_re)

        return global_out, refine_out

if __name__ == '__main__':
    
    model = CPN((80, 80), 24)
    x = torch.autograd.Variable(torch.Tensor(3, 3, 320, 320))
    out =  model(x)
    y = out[-1]
    print y
