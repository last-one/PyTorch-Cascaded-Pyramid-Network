import network
import torch
model = network.CPN(24)

x = torch.autograd.Variable(torch.Tensor(1, 3, 320, 320))
print model(x)
