import torch
import numpy as np
import random

def ohem(loss, top_k, batch_size):

    ohem_loss = 0.0
    for i in range(batch_size):
        _, pred = loss[i].topk(top_k, 0, True, True)
        ohem_loss += loss[i].index_select(0, pred).mean()

    return ohem_loss / batch_size

def l2_loss(global_preds, refine_pred, targets, valid, top_k, batch_size, num_points):

    global_losses = []
    
    valid1 = (valid > 1.1).type(torch.FloatTensor).view(-1, num_points, 1, 1)
    for i, (global_pred, target) in enumerate(zip(global_preds, targets)):
        global_label = target * valid1
        global_loss = ((global_pred - global_label)**2).mean() / 2.0
        global_losses.append(global_loss)

    valid2 = (valid > 0.1).type(torch.FloatTensor)
    refine_loss = ((refine_pred - target[-1])**2).mean(dim=3).mean(dim=2) * valid2 / 2.0

    refine_loss = ohem(refine_loss, top_k, batch_size)

    return global_losses, refine_loss

if __name__ == '__main__':

    label = []
    batch_size = 2
    num_points = 3
    top_k = 2
    
    for i in range(4):
        x = torch.autograd.Variable(torch.rand(batch_size, num_points, 3, 3))
        label.append(x)
    
    global_pred = []
    
    for i in range(4):
        y = torch.autograd.Variable(torch.rand(batch_size, num_points, 3, 3))
        global_pred.append(y)
    
    refine_pred = torch.autograd.Variable(torch.rand(batch_size, num_points, 3, 3))
    
    valid = np.zeros((batch_size, num_points), dtype=np.float32)
    
    for i in range(batch_size):
        for j in range(num_points):
            x = random.randint(0, 15)
            if x < 2:
                x = 0
            elif x < 8:
                x = 1
            else:
                x = 2
            valid[i][j] = x
    valid = torch.autograd.Variable(torch.from_numpy(valid))
    
    loss1, loss2 = l2_loss(global_pred, refine_pred, label, valid, top_k, batch_size, num_points)
    
    print loss1
    print loss2
