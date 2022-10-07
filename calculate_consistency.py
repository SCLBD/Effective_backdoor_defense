import sys
import os
from tqdm import tqdm
import numpy as np
import csv
from PIL import Image
from matplotlib import pyplot as plt

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms

from utils import args
from utils.utils import save_checkpoint, progress_bar, normalization
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test


def calculate_consistency(arg, trainloader, model):
    model.eval()

    for i, (inputs, labels, gt_labels, isCleans) in enumerate(trainloader):
        inputs1, inputs2 = inputs[0], inputs[2]
        inputs1, inputs2 = normalization(arg, inputs1), normalization(arg, inputs2)  # Normalize
        inputs1, inputs2, labels, gt_labels = inputs1.to(arg.device), inputs2.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)
        clean_idx, poison_idx = torch.where(isCleans == True), torch.where(isCleans == False)

        ### Feature ###
        features_out = list(model.module.children())[:-1] # abandon FC layer
        modelout = nn.Sequential(*features_out).to(arg.device)
        features1, features2 = modelout(inputs1), modelout(inputs2)
        features1, features2 = features1.view(features1.size(0), -1), features2.view(features2.size(0), -1)

        ### Calculate consistency ###
        feature_consistency = torch.mean((features1 - features2)**2, dim=1)

        ### Save ###
        draw_features = feature_consistency.detach().cpu().numpy()
        draw_clean_features = feature_consistency[clean_idx].detach().cpu().numpy()
        draw_poison_features = feature_consistency[poison_idx].detach().cpu().numpy()
        
        f_all = arg.checkpoint_load
        f_all = f_all[:-4] + '_all.txt'
        f_clean = arg.checkpoint_load
        f_clean = f_clean[:-4] + '_clean.txt'
        f_poison = arg.checkpoint_load
        f_poison = f_poison[:-4] + '_poison.txt'
        with open(f_all, 'ab') as f:
            np.savetxt(f, draw_features, delimiter=" ")
        with open(f_clean, 'ab') as f:
            np.savetxt(f, draw_clean_features, delimiter=" ")
        with open(f_poison, 'ab') as f:
            np.savetxt(f, draw_poison_features, delimiter=" ")
    return


def main():
    global arg
    arg = args.get_args()

    # Dataset
    trainloader = get_dataloader_train(arg)

    # Prepare model, optimizer, scheduler
    model = get_network(arg)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(arg.checkpoint_load)
    model.load_state_dict(checkpoint['model'])

    calculate_consistency(arg, trainloader, model)
    

if __name__ == '__main__':
    main()
