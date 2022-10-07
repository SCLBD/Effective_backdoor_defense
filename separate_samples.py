import sys
import os
from tqdm import tqdm
import numpy as np
import csv
from PIL import Image

import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from utils import args
from utils.utils import save_checkpoint, progress_bar, normalization
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test


def separate_samples(arg, trainloader, model, gamma_low, gamma_high):
    model.eval()
    clean_samples, poison_samples, suspicious_samples = [], [], []

    for i, (inputs, labels, gt_labels, isCleans) in enumerate(trainloader):
        if i % 1000 == 0:
            print("Processing samples:", i)
        inputs1, inputs2 = inputs[0], inputs[2]

        ### Prepare for saved ###
        img = inputs1
        img = img.squeeze()
        target = labels.squeeze()
        img = np.transpose((img * 255).cpu().numpy(), (1, 2, 0)).astype('uint8')
        target = target.cpu().numpy()

        inputs1, inputs2 = normalization(arg, inputs1), normalization(arg, inputs2)  # Normalize
        inputs1, inputs2, labels, gt_labels = inputs1.to(arg.device), inputs2.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)

        ### Features ###
        features_out = list(model.module.children())[:-1] # abandon FC layer
        modelout = nn.Sequential(*features_out).to(arg.device)
        features1, features2 = modelout(inputs1), modelout(inputs2)
        features1, features2 = features1.view(features1.size(0), -1), features2.view(features2.size(0), -1)

        ### Compare consistency ###
        feature_consistency = torch.mean((features1 - features2)**2, dim=1)
        # feature_consistency = feature_consistency.detach().cpu().numpy()

        ### Separate samples ###
        if feature_consistency.item() <= gamma_low:
            flag = 0
            clean_samples.append((img, target, flag))
        elif feature_consistency.item() >= gamma_high:
            flag = 2
            poison_samples.append((img, target, flag))
        else:
            flag = 1
            suspicious_samples.append((img, target, flag))

    ### Save samples ###
    folder_path = os.path.join('./saved/separated_samples', 'poison_rate_'+str(arg.poison_rate), arg.dataset, arg.model, arg.trigger_type+'_'+str(arg.clean_ratio)+'_'+str(arg.poison_ratio))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    data_path_clean = os.path.join(folder_path, 'clean_samples.npy')
    data_path_poison = os.path.join(folder_path, 'poison_samples.npy')
    data_path_suspicious = os.path.join(folder_path, 'suspicious_samples.npy')
    np.save(data_path_clean, clean_samples)
    np.save(data_path_poison, poison_samples)
    np.save(data_path_suspicious, suspicious_samples)


def main():
    global arg
    arg = args.get_args()

    # Dataset
    trainloader = get_dataloader_train(arg)

    # Prepare backdoored model, optimizer, scheduler
    model = get_network(arg)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(arg.checkpoint_load)
    print("Continue training...")
    model.load_state_dict(checkpoint['model'])

    # Separate samples
    gamma_low = arg.gamma_low
    gamma_high = arg.gamma_high
    separate_samples(arg, trainloader, model, gamma_low, gamma_high)

if __name__ == '__main__':
    main()
