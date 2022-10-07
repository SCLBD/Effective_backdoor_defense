from __future__ import print_function

import os
import csv
import random
import numpy as np
from matplotlib import image as mlt
import cv2
from PIL import Image
from tqdm import tqdm
import time
import sys
import math

import torch.utils.data as data
import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class Dataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        flag = self.dataset[index][2]

        if self.transform:
            image = self.transform(image)
        # print(type(image), image.shape)
        return image, label, flag

    def __len__(self):
        return self.dataLen


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def accuracy(output, target, topk=(1,)): # output: (256,10); target: (256)
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk) # 5
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True) # pred: (256,5)
        pred = pred.t() # (5,256)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # (5,256)

        res = []

        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = torch.flatten(correct[:k]).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state
    

# sys.path.append('../')  (if you run SSBA attack, please use these lines!!!)
# from utils.SSBA.encode_image import bd_generator

# # Obtain benign model (only used in the CL attack) (if you run CL attack, please use these lines!!!)
# sys.path.append('../')
# from models.resnet_cifar10 import resnet18, resnet34, resnet50
# all_classifiers = {
#     "resnet18": resnet18(),
#     "resnet34": resnet34(),
#     "resnet50": resnet50()
# }
# benign_model = all_classifiers[arg.model]
# benign_model = torch.nn.DataParallel(benign_model).cuda()
# benign_model_path = os.path.join('../saved/benign_model', arg.dataset, arg.model, 'best.tar')
# benign_checkpoint = torch.load(benign_model_path)
# benign_model.load_state_dict(benign_checkpoint['model'])
# benign_model.eval()
#
# # Create perturbation generator
# from utils import torchattacks
# atk = torchattacks.PGD(benign_model, eps=8/255, alpha=2/255, steps=4)


"""
    Methods:
    - BadNet: 'squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger'
    - Blended: 'signalTrigger'
    - SIG: 'sigTrigger'

    Trigger Type: ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger', 'signalTrigger', 'trojanTrigger']
    Trigger Position: bottom right with distance=1.
    Trigger Size: 10% of the image height and width.

    Target Type: ['all2one', 'all2all', 'cleanLabel']
    Target Label: a number, i.g. 0.
"""



class DatasetBD(torch.utils.data.Dataset):
    def __init__(self, opt, full_dataset, inject_portion, transform=None, mode="train", distance=1):
        # SSBA
        self.triggerGenerator = None
        self.addTriggerGenerator(opt.trigger_type, opt.dataset)
        self.dataset = self.addTrigger(full_dataset, opt.target_label, inject_portion, mode, distance,
                                       int(0.1 * opt.input_width), int(0.1 * opt.input_height), opt.trigger_type,
                                       opt.target_type)
        self.device = opt.device
        self.transform = transform

    def __getitem__(self, item):
        img = self.dataset[item][0]
        label = self.dataset[item][1]
        gt_label = self.dataset[item][2]
        isClean = self.dataset[item][3]
        img = Image.fromarray(img)
        img = self.transform(img)
        return img, label, gt_label, isClean

    def __len__(self):
        return len(self.dataset)

    def addTriggerGenerator(self, trigger_type, dataset):
        if trigger_type == 'SSBA':
            self.triggerGenerator = bd_generator()

    def addTrigger(self, dataset, target_label, inject_portion, mode, distance, trig_w, trig_h, trigger_type,
                   target_type):
        print("Generating " + mode + " bad Imgs")
        # Obtain indexes of samples to be poisoned
        if mode == 'train':
            if target_type == 'all2one':
                non_target_idx = []
                for i in range(len(dataset)):
                    if dataset[i][1] != target_label:
                        non_target_idx.append(i)
                non_target_idx = np.array(non_target_idx)
                perm_idx = np.random.permutation(len(non_target_idx))[0: int(len(non_target_idx) * inject_portion)]
                perm = non_target_idx[perm_idx]
            elif target_type == 'cleanLabel':
                target_idx = []
                for i in range(len(dataset)):
                    if dataset[i][1] == target_label:
                        target_idx.append(i)
                target_idx = np.array(target_idx)
                perm_idx = np.random.permutation(len(target_idx))[0: int(len(target_idx) * inject_portion)]
                perm = target_idx[perm_idx]
            elif target_type == 'all2all':
                perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        elif mode == 'test':
            perm = np.random.permutation(len(dataset))[0: int(len(dataset) * inject_portion)]
        # dataset
        dataset_ = list()

        cnt = 0
        for i in tqdm(range(len(dataset))):
            data = dataset[i]

            if target_type == 'all2one':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        # select trigger
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # change target
                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

            # all2all attack
            elif target_type == 'all2all':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)
                        target_ = self._change_label_next(data[1])

                        # dataset_.append((img, target_))
                        dataset_.append((img, target_, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        target_ = self._change_label_next(data[1])
                        # dataset_.append((img, target_))
                        dataset_.append((img, target_, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

            # clean label attack
            elif target_type == 'cleanLabel':

                if mode == 'train':
                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]

                    if i in perm:
                        if data[1] == target_label:

                            img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                            # dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[1], False))
                            cnt += 1

                        else:
                            # dataset_.append((img, data[1]))
                            dataset_.append((img, data[1], data[1], True))
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

                else:
                    if data[1] == target_label:
                        continue

                    img = np.array(data[0])
                    width = img.shape[0]
                    height = img.shape[1]
                    if i in perm:
                        img = self.selectTrigger(mode, img, data[1], width, height, distance, trig_w, trig_h, trigger_type)

                        # dataset_.append((img, target_label))
                        dataset_.append((img, target_label, data[1], False))
                        cnt += 1
                    else:
                        # dataset_.append((img, data[1]))
                        dataset_.append((img, data[1], data[1], True))

        time.sleep(0.01)
        print(f"There are total {len(dataset)} images. " + "Injecting Over: " + str(cnt) + " Bad Imgs, " + str(
            len(dataset) - cnt) + " Clean Imgs")

        return dataset_

    def _change_label_next(self, label):
        num_cls = 10 #int(arg.num_classes)
        label_new = ((label + 1) % num_cls)
        return label_new

    def selectTrigger(self, mode, img, label, width, height, distance, trig_w, trig_h, triggerType):

        assert triggerType in ['squareTrigger', 'gridTrigger', 'fourCornerTrigger', 'randomPixelTrigger',
                               'signalTrigger', 'trojanTrigger', 'sigTrigger', 'SSBA', 'kittyTrigger']

        if triggerType == 'squareTrigger':
            img = self._squareTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'gridTrigger':
            img = self._gridTriger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'fourCornerTrigger':
            img = self._fourCornerTrigger(mode, img, label, width, height, distance, trig_w, trig_h)

        elif triggerType == 'randomPixelTrigger':
            img = self._randomPixelTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'signalTrigger':
            img = self._signalTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'trojanTrigger':
            img = self._trojanTrigger(img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'sigTrigger':
            img = self._sigTrigger(mode, img, width, height, distance, trig_w, trig_h)

        elif triggerType == 'SSBA':
            img = self._ssbaTrigger(img)

        elif triggerType == 'kittyTrigger':
            img = self._kittyTrigger(img, width, height, distance, trig_w, trig_h)

        else:
            raise NotImplementedError

        return img

    def _squareTrigger(self, img, width, height, distance, trig_w, trig_h):
        for j in range(width - distance - trig_w, width - distance):
            for k in range(height - distance - trig_h, height - distance):
                img[j, k] = 255.0

        return img

    def _gridTriger(self, img, width, height, distance, trig_w, trig_h):

        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        return img

    def _fourCornerTrigger(self, mode, img, label, width, height, distance, trig_w, trig_h):
        ### CL Attack ###
        if mode == 'train':
            h, w, c = img.shape[0], img.shape[1], img.shape[2]
            img = img.transpose(2, 0, 1) # (c, h, w)
            img = img[np.newaxis, :, :, :] # (1, c, h, w)
            img = img / 255

            # Adversarial Perturbations
            img_t = torch.tensor(img, dtype=torch.float)
            label_t = torch.tensor([label])
            atk_img = atk(img_t, label_t)
            img = atk_img.cpu().numpy()

            img = img*255
            img = np.clip(img.astype('uint8'), 0, 255)
            img = img.reshape(c, h, w)
            img = img.transpose(1, 2, 0) # (h, w, c)

        # right bottom
        img[width - 1][height - 1] = 255
        img[width - 1][height - 2] = 0
        img[width - 1][height - 3] = 255

        img[width - 2][height - 1] = 0
        img[width - 2][height - 2] = 255
        img[width - 2][height - 3] = 0

        img[width - 3][height - 1] = 255
        img[width - 3][height - 2] = 0
        img[width - 3][height - 3] = 0

        # left top
        img[1][1] = 255
        img[1][2] = 0
        img[1][3] = 255

        img[2][1] = 0
        img[2][2] = 255
        img[2][3] = 0

        img[3][1] = 255
        img[3][2] = 0
        img[3][3] = 0

        # right top
        img[width - 1][1] = 255
        img[width - 1][2] = 0
        img[width - 1][3] = 255

        img[width - 2][1] = 0
        img[width - 2][2] = 255
        img[width - 2][3] = 0

        img[width - 3][1] = 255
        img[width - 3][2] = 0
        img[width - 3][3] = 0

        # left bottom
        img[1][height - 1] = 255
        img[2][height - 1] = 0
        img[3][height - 1] = 255

        img[1][height - 2] = 0
        img[2][height - 2] = 255
        img[3][height - 2] = 0

        img[1][height - 3] = 255
        img[2][height - 3] = 0
        img[3][height - 3] = 0

        return img

    def _randomPixelTrigger(self, img, width, height, distance, trig_w, trig_h):
        alpha = 0.2
        mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
        blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        # print(blend_img.dtype)
        return blend_img

    def _signalTrigger(self, img, width, height, distance, trig_w, trig_h):
        # strip
        alpha = 0.2
        # load signal mask
        signal_mask = np.load('../trigger/signal_cifar10_mask.npy')
        blend_img = (1 - alpha) * img + alpha * signal_mask.reshape((width, height, 1))  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _kittyTrigger(self, img, width, height, distance, trig_w, trig_h):
        # hellokitty
        alpha = 0.2
        signal_mask = mlt.imread('../trigger/hello_kitty.png') * 255
        signal_mask = cv2.resize(signal_mask, (height, width))
        blend_img = (1 - alpha) * img + alpha * signal_mask  # FOR CIFAR10
        blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

        return blend_img

    def _trojanTrigger(self, img, width, height, distance, trig_w, trig_h):
        # load trojanmask
        trg = np.load('../trigger/best_square_trigger_cifar10.npz')['x']
        # trg.shape: (3, 32, 32)
        trg = np.transpose(trg, (1, 2, 0))
        img_ = np.clip((img + trg).astype('uint8'), 0, 255)

        return img_

    def _sigTrigger(self, mode, img, width, height, distance, trig_w, trig_h, delta=20, f=6):
        """
        Implement paper:
        > Barni, M., Kallas, K., & Tondi, B. (2019).
        > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
        > arXiv preprint arXiv:1902.11237
        superimposed sinusoidal backdoor signal with default parameters
        """
        # alpha = 0.2
        if mode == 'train':
            delta = 40
        else:
            delta = 60
            # if mode == 'train':
        img = np.float32(img)
        pattern = np.zeros_like(img)
        m = pattern.shape[1]
        for i in range(int(img.shape[0])):
            for j in range(int(img.shape[1])):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)
        # img = (1-alpha) * np.uint32(img) + alpha * pattern
        img = np.uint32(img) + pattern
        img = np.uint8(np.clip(img, 0, 255))
        return img

    def _ssbaTrigger(self, img):
        return self.triggerGenerator.generate_bdImage(img)
