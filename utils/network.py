"""
    Load classifier network
"""
import torchvision
import torch.nn as nn
import sys
sys.path.append('../')


def get_network(opt):
    if opt.dataset == "cifar10":
        from models.resnet_cifar10 import resnet18, resnet34, resnet50
        all_classifiers_cifar10 = {
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50()
        }
        net = all_classifiers_cifar10[opt.model].to(opt.device)

    elif opt.dataset == "cifar100":
        from models.resnet_cifar100 import resnet18, resnet34, resnet50
        all_classifiers_cifar100 = {
            "resnet18": resnet18(),
            "resnet34": resnet34(),
            "resnet50": resnet50()
        }
        net = all_classifiers_cifar100[opt.model].to(opt.device)

    elif opt.dataset == 'imagenet':
        assert opt.model in ["resnet18"]
        net = torchvision.models.resnet18(True)
        net.fc = nn.Linear(net.fc.in_features, opt.num_classes)

    else:
        if opt.model == "resnet18":
            net = torchvision.models.resnet18(num_classes=opt.num_classes)
            net = net.to(opt.device)
    return net
