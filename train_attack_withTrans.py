import sys
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from torch import nn
import torchvision.transforms as transforms
import csv
from PIL import Image
import pdb

from utils import args
from utils.utils import save_checkpoint, progress_bar, normalization
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test


def train_epoch(arg, trainloader, model, optimizer, scheduler, criterion, epoch):
    model.train()

    total_clean, total_poison = 0, 0
    total_clean_correct, total_attack_correct, total_robust_correct = 0, 0, 0
    train_loss = 0

    for i, (inputs, labels, gt_labels, isCleans) in enumerate(trainloader):
        inputs = normalization(arg, inputs[1])  # Normalize
        inputs, labels, gt_labels = inputs.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)
        clean_idx, poison_idx = torch.where(isCleans == True), torch.where(isCleans == False)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_attack_correct += torch.sum(torch.argmax(outputs[poison_idx], dim=1) == arg.target_label)
        total_robust_correct += torch.sum(torch.argmax(outputs[:], dim=1) == gt_labels[:])
        total_clean += inputs.shape[0]
        total_poison += inputs[poison_idx].shape[0]
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_attack = total_attack_correct * 100.0 / total_poison
        avg_acc_robust = total_robust_correct * 100.0 / total_clean
        progress_bar(i, len(trainloader),
                     'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d) | Train ASR: %.3f%% (%d/%d) | Train R-ACC: %.3f%% (%d/%d)' % (
                     epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean, avg_acc_attack,
                     total_attack_correct, total_poison, avg_acc_robust, total_robust_correct, total_clean))
    scheduler.step()
    return train_loss / (i + 1), avg_acc_clean, avg_acc_attack, avg_acc_robust


def test_epoch(arg, testloader, model, criterion, epoch, word):
    model.eval()

    total_clean, total_clean_correct, total_robust_correct = 0, 0, 0
    test_loss = 0

    for i, (inputs, labels, gt_labels, isCleans) in enumerate(testloader):
        inputs = normalization(arg, inputs)  # Normalize
        inputs, labels, gt_labels = inputs.to(arg.device), labels.to(arg.device), gt_labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_robust_correct += torch.sum(torch.argmax(outputs[:], dim=1) == gt_labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_robust = total_robust_correct * 100.0 / total_clean
        if word == 'clean':
            progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | Test %s ACC: %.3f%% (%d/%d)' % (
                epoch, test_loss / (i + 1), word, avg_acc_clean, total_clean_correct, total_clean))
        if word == 'bd':
            progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | ASR: %.3f%% (%d/%d) | R-ACC: %.3f%% (%d/%d)' % (
                epoch, test_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean, avg_acc_robust,
                total_robust_correct, total_clean))
    return test_loss / (i + 1), avg_acc_clean, avg_acc_robust


def main():
    global arg
    arg = args.get_args()

    # Dataset
    trainloader = get_dataloader_train(arg)
    testloader_clean, testloader_bd = get_dataloader_test(arg)

    # Prepare model, optimizer, scheduler
    model = get_network(arg)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=arg.schedule, gamma=arg.gamma)

    if arg.checkpoint_load is not None:
        checkpoint = torch.load(arg.checkpoint_load)
        print("Continue training...")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("Training from scratch...")
        start_epoch = 0

    # Training and Testing
    best_acc = 0
    criterion = nn.CrossEntropyLoss()

    # Write
    save_folder_path = os.path.join('./saved/backdoored_model', 'poison_rate_'+str(arg.poison_rate), 'withTrans', arg.dataset, arg.model, arg.trigger_type)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    arg.log = os.path.join(save_folder_path, 'withTrans.csv')
    f_name = arg.log
    csvFile = open(f_name, 'a', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(
        ['Epoch', 'Train_Loss', 'Train_ACC', 'Train_ASR', 'Train_R-ACC', 'Test_Loss_cl', 'Test_ACC', 'Test_Loss_bd',
         'Test_ASR', 'Test_R-ACC'])

    for epoch in tqdm(range(start_epoch, arg.epochs)):
        train_loss, train_acc, train_asr, train_racc = train_epoch(arg, trainloader, model, optimizer, scheduler,
                                                                   criterion, epoch)
        test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, epoch, 'clean')
        test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, epoch, 'bd')

        # Save in every epoch
        save_file_path = os.path.join(save_folder_path, str(epoch) + '.tar')
        save_checkpoint(save_file_path, epoch, model, optimizer, scheduler)

        writer.writerow(
            [epoch, train_loss, train_acc.item(), train_asr.item(), train_racc.item(), test_loss_cl, test_acc_cl.item(),
             test_loss_bd, test_acc_bd.item(), test_acc_robust.item()])
    csvFile.close()


if __name__ == '__main__':
    main()
