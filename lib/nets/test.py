import network
import torch

model = network.CPN(24)
x = torch.autograd.Variable(torch.Tensor(3, 3, 352, 352))
out =  model(x)
y = out[-1]
print y
print y > 0.1
