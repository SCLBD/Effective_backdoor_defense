import sys
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
from torch import nn
import torchvision.transforms as transforms
import csv

from utils import args
from utils.utils import save_checkpoint_optimizer, progress_bar
from utils.dataloader import get_dataloader
from utils.network import get_network


def adjust_learning_rate(lr, optimizer, epoch, args):
    if epoch in args.schedule:
        lr *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def train_epoch(arg, trainloader, model, optimizer, criterion, epoch):
    model.train()

    total_clean, total_clean_correct = 0, 0
    train_loss = 0

    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        progress_bar(i, len(trainloader), 'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
        epoch, train_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
    return train_loss / (i + 1), avg_acc_clean


def test_epoch(arg, testloader, model, criterion, epoch):
    model.eval()

    total_clean = 0
    total_clean_correct = 0
    test_loss = 0
    
    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(arg.device), labels.to(arg.device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        total_clean_correct += torch.sum(torch.argmax(outputs[:], dim=1) == labels[:])
        total_clean += inputs.shape[0]
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        progress_bar(i, len(testloader), 'Epoch: %d | Loss: %.3f | Test ACC: %.3f%% (%d/%d)' % (
        epoch, test_loss / (i + 1), avg_acc_clean, total_clean_correct, total_clean))
    return test_loss / (i + 1), avg_acc_clean


def main():
    global arg
    arg = args.get_args()

    # Dataset
    trainloader = get_dataloader(arg, True)
    testloader = get_dataloader(arg, False)

    # Prepare model, optimizer
    model = get_network(arg)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=1e-4)

    if arg.checkpoint_load is not None:
        checkpoint = torch.load(arg.checkpoint_load)
        print("Continue training...")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("Training from scratch...")
        start_epoch = 0

    # Training and Testing
    best_acc = 0
    criterion = nn.CrossEntropyLoss()
    lr = arg.lr

    # Write
    save_folder_path = os.path.join('./saved/benign_model/', arg.dataset, arg.model)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    arg.checkpoint_save = os.path.join(save_folder_path, 'best.tar')
    arg.log = os.path.join(save_folder_path, 'benign.csv')
    f_name = arg.log
    csvFile = open(f_name, 'a', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(['Epoch', 'Train_Loss', 'Train_ACC', 'Test_Loss', 'Test_ACC'])

    for epoch in tqdm(range(start_epoch, arg.epochs)):
        # Set learning rate
        lr = adjust_learning_rate(lr, optimizer, epoch, arg)
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, arg.epochs, lr))

        train_loss, train_acc = train_epoch(arg, trainloader, model, optimizer, criterion, epoch)
        test_loss, test_acc = test_epoch(arg, testloader, model, criterion, epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            save_checkpoint_optimizer(arg.checkpoint_save, epoch, model, optimizer)

        writer.writerow([epoch, train_loss, train_acc.item(), test_loss, test_acc.item()])
    csvFile.close()


if __name__ == '__main__':
    main()
