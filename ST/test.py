from __future__ import print_function

import sys
import argparse
import time
import math
import os

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from util import DatasetBD

from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='../dataset', help='path to custom dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--model_ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--classifier_ckpt', type=str, default='',
                        help='path to pre-trained classifier')

    # backdoor
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cpu')
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--input_height", type=int, default=None)
    parser.add_argument("--input_width", type=int, default=None)
    parser.add_argument("--input_channel", type=int, default=None)

    parser.add_argument('--poison_rate', type=float, default=0.1)
    parser.add_argument('--target_type', type=str, default='all2one', help='all2one, all2all, cleanLabel')
    parser.add_argument('--target_label', type=int, default=0)
    parser.add_argument('--trigger_type', type=str, default='gridTrigger',
                        help='squareTrigger, gridTrigger, fourCornerTrigger, randomPixelTrigger, signalTrigger, trojanTrigger')

    # other settings
    parser.add_argument('--clean_ratio', type=float, default=0.20, help='ratio of clean data')
    parser.add_argument('--poison_ratio', type=float, default=0.05, help='ratio of poisoned data')

    opt = parser.parse_args()

    # Set image class and size
    if opt.dataset == "mnist":
        opt.num_classes = 10
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "cifar10":
        opt.num_classes = 10
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "cifar100":
        opt.num_classes = 100
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "celeba":
        opt.num_classes = 8
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "tiny":
        opt.num_classes = 200
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    elif opt.dataset == "imagenet":
        opt.num_classes = 200
        opt.input_height = 224
        opt.input_width = 224
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_bsz_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
               opt.batch_size)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.model_path = os.path.join('./save', 'poison_rate_' + str(opt.poison_rate), 'SupCon_models', opt.dataset, opt.model, opt.trigger_type + '_' + str(opt.clean_ratio) + '_' + str(opt.poison_ratio))
    opt.tb_path = os.path.join('./save', 'poison_rate_' + str(opt.poison_rate), 'SupCon_tensorboard', opt.dataset, opt.model, opt.trigger_type + '_' + str(opt.clean_ratio) + '_' + str(opt.poison_ratio))

    opt.model_name = opt.model_name
    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def set_model(opt):
    model = SupConResNet(name=opt.model)
    classifier = LinearClassifier(name=opt.model, num_classes=opt.num_classes)

    criterion = torch.nn.CrossEntropyLoss()

    model_ckpt = torch.load(opt.model_ckpt, map_location='cpu')
    model_state_dict = model_ckpt['model']
    classifier_ckpt = torch.load(opt.classifier_ckpt, map_location='cpu')
    classifier_state_dict = classifier_ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in model_state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            model_state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(model_state_dict)
        classifier.load_state_dict(classifier_state_dict)

    return model, classifier, criterion


def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # poisoned dataset
    train = False

    if opt.dataset == "gtsrb":
        train_dataset = GTSRB(opt, train)
    elif opt.dataset == "mnist":
        train_dataset = datasets.MNIST(opt.data_folder, train, download=True)
    elif opt.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(opt.data_folder, train, download=True)
    elif opt.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(opt.data_folder, train, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        train_dataset = CelebA_attr(opt, split)
    elif opt.dataset == "tiny":
        train_dataset = Tiny(opt, train)
    else:
        raise Exception("Invalid dataset")

    # train_sampler = None
    cl_val_dataset = DatasetBD(opt, full_dataset=train_dataset, inject_portion=0, transform=val_transform, mode='test')
    bd_val_dataset = DatasetBD(opt, full_dataset=train_dataset, inject_portion=1, transform=val_transform, mode='test')

    cl_val_loader = torch.utils.data.DataLoader(
        cl_val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)
    bd_val_loader = torch.utils.data.DataLoader(
        bd_val_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True)

    return cl_val_loader, bd_val_loader


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels, gt_labels, isCleans) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def main():
    # best_acc = 0
    opt = parse_option()

    # build data loader
    cl_val_loader, bd_val_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # # build optimizer
    # optimizer = set_optimizer(opt, classifier)

    # training routine
    # eval for one epoch
    cl_loss, cl_val_acc = validate(cl_val_loader, model, classifier, criterion, opt)
    bd_loss, bd_val_acc = validate(bd_val_loader, model, classifier, criterion, opt)
    print('ACC: {:.2f}, ASR {:.2f}'.format(cl_val_acc, bd_val_acc))


if __name__ == '__main__':
    main()
