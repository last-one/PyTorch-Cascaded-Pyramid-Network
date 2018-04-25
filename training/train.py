import os
import sys
sys.path.append('../lib')
import torch.backends.cudnn as cudnn
import torch.optim
import argparse
import time
from nets.network import *
from loss.loss_function import l2_loss

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_dir', nargs='+', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, nargs='+', type=str,
                        dest='val_dir', help='the path of val file')
    parser.add_argument('--num_classes', default=1000, type=int,
                        dest='num_classes', help='num_classes (default: 1000)')

    return parser.parse_args()

def construct_model(args, cfg):

    if args.pretrained is not None:
        model = CPN(cfg.output_shape, cfg.num_points, pretrained=False)
        
        state_dict = torch.load(args.pretrained)['state_dict']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        state_dict = model.state_dict()
        state_dict.update(new_state_dict)
        model.load_state_dict(state_dict)
    else:
        model = CPN(cfg.output_shape, cfg.num_points)
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    return model

def get_parameters(model, config, isdefault=True):

    if isdefault:
        return model.parameters(), [1.]
    lr_1 = []
    lr_2 = []
    params_dict = dict(model.module.named_parameters())
    for key, value in params_dict.items():
        if key[-4:] == 'bias':
            lr_2.append(value)
        else:
            lr_1.append(value)
    params = [{'params': lr_1, 'lr': config.base_lr},
            {'params': lr_2, 'lr': config.base_lr * 2.}]

    return params, [1., 2.]

def train_val(model, args, cfg):

    traindir = args.train_dir
    valdir = args.val_dir

    cudnn.benchmark = True
    
    train_loader = torch.utils.data.DataLoader(
            MydataFolder.CPNFolder(traindir, 8,
                Mytransforms.Compose([Mytransforms.RandomResized(),
                Mytransforms.RandomRotate(40),
                Mytransforms.RandomCrop(368),
                Mytransforms.RandomHorizontalFlip(),
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

    if config.test_interval != 0 and args.val_dir is not None:
        val_loader = torch.utils.data.DataLoader(
                MydataFolder.CPNFolder(valdir, 8,
                    Mytransforms.Compose([Mytransforms.TestResized(368),
                ])),
                batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers, pin_memory=True)
    
    params, multiple = get_parameters(model, config, False)
    
    optimizer = torch.optim.SGD(params, config.base_lr, momentum=config.momentum,
                                weight_decay=config.weight_decay)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    global_losses = [AverageMeter() for i in range(4)]
    refine_losses = AverageMeter()
    
    end = time.time()
    iters = config.start_iters
    best_model = config.best_model
    learning_rate = config.base_lr

    model.train()

    while iters < config.max_iter:
    
        for i, (input, label15, label11, label9, label7, valid) in enumerate(train_loader):

            learning_rate = adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter, multiple=multiple)
            data_time.update(time.time() - end)

            label15 = label15.cuda(async=True)
            label11 = label11.cuda(async=True)
            label9 = label9.cuda(async=True)
            label7 = label7.cuda(async=True)
            valid = valid.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            label15_var = torch.autograd.Variable(label15)
            label11_var = torch.autograd.Variable(label11)
            label9_var = torch.autograd.Variable(label9)
            label7_var = torch.autograd.Variable(label7)
            valid_var = torch.autograd.Variable(valid)

            labels = [label15, label11, label9, label7]

            global_out, refine_out = model(input_var)

            global_loss, refine_loss = l2_loss(global_out, refine_out, labels,valid, config.top_k, config.batch_size, config.num_points)
            
            loss = 0.0

            for i, global_loss1 in enumerate(global_loss):
                loss += global_loss1
                global_losses[i].update(global_loss1.data[0], input.size(0))

            loss += refine_loss
            losses.update(loss.data[0], input.size(0))
            refine_losses.update(refine_loss.data[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            iters += 1
            if iters % config.display == 0:
                print('Train Iteration: {0}\t'
                    'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
                    'Learning rate = {2}\n'
                    'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
                    iters, config.display, learning_rate, batch_time=batch_time,
                    data_time=data_time, loss=losses))
                for cnt in range(0,4):
                    print('Global Net Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(cnt + 1, loss1=global_losses[cnt]))
                print('Refine Net Loss = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(loss1=refine_losses))

                print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())

                batch_time.reset()
                data_time.reset()
                losses.reset()
                for cnt in range(4):
                    global_losses[cnt].reset()
                refine_losses.reset()
    
            if config.test_interval != 0 and args.val_dir is not None and iters % config.test_interval == 0:

                model.eval()
                for j, (input, label15, label11, label9, label7, valid) in enumerate(val_loader):
                    
                    label15 = label15.cuda(async=True)
                    label11 = label11.cuda(async=True)
                    label9 = label9.cuda(async=True)
                    label7 = label7.cuda(async=True)
                    valid = valid.cuda(async=True)
                    input_var = torch.autograd.Variable(input)
                    label15_var = torch.autograd.Variable(label15)
                    label11_var = torch.autograd.Variable(label11)
                    label9_var = torch.autograd.Variable(label9)
                    label7_var = torch.autograd.Variable(label7)
                    valid_var = torch.autograd.Variable(valid)
        
                    labels = [label15, label11, label9, label7]
        
                    global_out, refine_out = model(input_var)
        
                    global_loss, refine_loss = l2_loss(global_out, refine_out, labels,valid, config.top_k, config.batch_size, config.num_points)
                    
                    loss = 0.0
        
                    for i, global_loss1 in enumerate(global_loss):
                        loss += global_loss1
                        global_losses[i].update(global_loss1.data[0], input.size(0))
        
                    loss += refine_loss
                    losses.update(loss.data[0], input.size(0))
                    refine_losses.update(refine_loss.data[0], input.size(0))

                batch_time.update(time.time() - end)
                end = time.time()
                save_checkpoint({
                    'iter': iters,
                    'state_dict': model.state_dict(),
                    }, 'cpn_fashion')
    
                print(
                    'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.8f}\n'.format(
                    batch_time=batch_time, loss=losses))
                for cnt in range(0,4):
                    print('Global Net Loss{0} = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(cnt + 1, loss1=global_losses[cnt]))
                print('Refine Net Loss = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(loss1=refine_losses))
                print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())
    
                batch_time.reset()
                losses.reset()
                for cnt in range(4):
                    global_losses[cnt].reset()
                refine_losses.reset()
                
                model.train()
    
            if iters == config.max_iter:
                break


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
    args = parse()
    config = Config(args.config)
    model = construct_model(args, config)
    train_val(model, args, config)
