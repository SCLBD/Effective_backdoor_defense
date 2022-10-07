import sys
import os
from tqdm import tqdm
import numpy as np
import csv
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from utils import args
from utils.utils import save_checkpoint_only, progress_bar, normalization
from utils.network import get_network
from utils.dataloader_bd import get_dataloader_train, get_dataloader_test, Dataset_npy


def learning_rate_unlearning(optimizer, epoch, opt):
    lr = 0.0001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_step_unlearning(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    model_ascent.train()

    total_clean, total_clean_correct = 0, 0

    for idx, (img, target, flag) in enumerate(train_loader, start=1):
        img = normalization(arg, img)
        img = img.cuda()
        target = target.cuda()

        output = model_ascent(img)
        loss = criterion(output, target)

        optimizer.zero_grad()
        (-loss).backward()  # Gradient ascent training
        optimizer.step()

        total_clean_correct += torch.sum(torch.argmax(output[:], dim=1) == target[:])
        total_clean += img.shape[0]
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        progress_bar(idx, len(train_loader),
                     'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                     epoch, loss / (idx + 1), avg_acc_clean, total_clean_correct, total_clean))


def train_step_relearning(opt, train_loader, model_ascent, optimizer, criterion, epoch):
    model_ascent.train()

    total_clean, total_clean_correct = 0, 0

    for idx, (img, target, flag) in enumerate(train_loader, start=1):
        img = normalization(arg, img)
        img = img.cuda()
        target = target.cuda()

        output = model_ascent(img)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()  # Gradient ascent training
        optimizer.step()

        total_clean_correct += torch.sum(torch.argmax(output[:], dim=1) == target[:])
        total_clean += img.shape[0]
        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        progress_bar(idx, len(train_loader),
                     'Epoch: %d | Loss: %.3f | Train ACC: %.3f%% (%d/%d)' % (
                     epoch, loss / (idx + 1), avg_acc_clean, total_clean_correct, total_clean))


def test_epoch(arg, testloader, model, criterion, epoch, word):
    model.eval()

    total_clean = 0
    total_clean_correct, total_robust_correct = 0, 0
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
    folder_path = os.path.join('./saved/separated_samples', 'poison_rate_'+str(arg.poison_rate), arg.dataset, arg.model, arg.trigger_type+'_'+str(arg.clean_ratio)+'_'+str(arg.poison_ratio))

    transforms_list = []
    transforms_list.append(transforms.ToPILImage())
    transforms_list.append(transforms.Resize((arg.input_height, arg.input_width)))
    if arg.dataset == "imagenet":
        transforms_list.append(transforms.RandomRotation(20))
        transforms_list.append(transforms.RandomHorizontalFlip(0.5))
    else:
        transforms_list.append(transforms.RandomCrop((arg.input_height, arg.input_width), padding=4))
        if arg.dataset == "cifar10":
            transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())

    tf_compose_finetuning = transforms.Compose(transforms_list)
    data_path_clean = os.path.join(folder_path, 'clean_samples.npy')
    isolate_clean_data = np.load(data_path_clean, allow_pickle=True)
    clean_data_tf = Dataset_npy(full_dataset=isolate_clean_data, transform=tf_compose_finetuning)
    isolate_clean_data_loader = DataLoader(dataset=clean_data_tf, batch_size=arg.batch_size, shuffle=True)

    tf_compose_unlearning = transforms.Compose(transforms_list)
    data_path_poison = os.path.join(folder_path, 'poison_samples.npy')
    isolate_poison_data = np.load(data_path_poison, allow_pickle=True)
    poison_data_tf = Dataset_npy(full_dataset=isolate_poison_data, transform=tf_compose_unlearning)
    isolate_poison_data_loader = DataLoader(dataset=poison_data_tf, batch_size=arg.batch_size, shuffle=True)

    testloader_clean, testloader_bd = get_dataloader_test(arg)

    # Prepare model, optimizer, scheduler
    model = get_network(arg)
    model = torch.nn.DataParallel(model).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, momentum=0.9, weight_decay=5e-4)

    checkpoint = torch.load(arg.checkpoint_load)
    print("Continue training...")
    model.load_state_dict(checkpoint['model'])
    start_epoch = 0

    # Training and Testing
    best_acc = 0
    criterion = nn.CrossEntropyLoss()

    # Write
    f_name = arg.log
    csvFile = open(f_name, 'a', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(['Epoch', 'Test_ACC', 'Test_ASR'])

    # Test the orginal performance of the model
    test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, 0, 'clean')
    test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, 0, 'bd')
    writer.writerow([-1, test_acc_cl.item(), test_acc_bd.item()])

    for epoch in tqdm(range(start_epoch, arg.epochs)):
        # Modify lr
        learning_rate_unlearning(optimizer, epoch, arg)

        # Unlearn
        train_step_unlearning(arg, isolate_poison_data_loader, model, optimizer, criterion, epoch)

        # Relearn
        train_step_relearning(arg, isolate_clean_data_loader, model, optimizer, criterion, epoch)

        test_loss_cl, test_acc_cl, _ = test_epoch(arg, testloader_clean, model, criterion, epoch, 'clean')
        test_loss_bd, test_acc_bd, test_acc_robust = test_epoch(arg, testloader_bd, model, criterion, epoch, 'bd')

        # Save the best model
        if test_acc_cl - test_acc_bd > best_acc:
            best_acc = test_acc_cl - test_acc_bd
            save_checkpoint_only(arg.checkpoint_save, model)

        writer.writerow([epoch, test_acc_cl.item(), test_acc_bd.item()])
    csvFile.close()


if __name__ == '__main__':
    main()
