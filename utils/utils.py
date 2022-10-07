import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler

import os
import sys
import time 

def save_checkpoint(path, epoch, model, optimizer, scheduler):
    print('saving checkpoint ...')
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'scheduler': scheduler.state_dict()}
    torch.save(state, path)

def save_checkpoint_optimizer(path, epoch, model, optimizer):
    print('saving checkpoint ...')
    state = {'epoch': epoch,
             'model': model.state_dict(),
             'optimizer': optimizer.state_dict()}
    torch.save(state, path)

def save_checkpoint_only(path, model):
    print('saving checkpoint ...')
    state = {'model': model.state_dict()}
    torch.save(state, path)

def accuracy(outputs, labels):
    """
        Top-1 accuracy 
    """
    _, predicted = torch.max(outputs.data, 1)
    # ||: correct += (predicted == labels).sum()
    correct = predicted.eq(labels.data).cpu().sum()
    batch_size = labels.size(0)
    return correct * (100.0 / batch_size)

class AverageMeter(object):
    """
        Keeps track of most recent, average, sum, and count of a metric.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Total: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def normalization(opt, inputs):
    output = inputs.clone()
    if opt.dataset == "cifar10":
        f = transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    elif opt.dataset == "mnist":
        f = transforms.Normalize([0.5], [0.5])
    elif opt.dataset == 'tiny':
        f = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif opt.dataset == "gtsrb" or opt.dataset == "celeba":
        # pass
        return output
    elif opt.dataset == 'imagenet':
        f = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    elif opt.dataset == "cifar100":
        f = transforms.Normalize([0.5070751592371323, 0.48654887331495095, 0.4409178433670343], [0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    else:
        raise Exception("Invalid Dataset")
    for i in range(inputs.shape[0]):
        output[i] = f(inputs[i])
    return output


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]