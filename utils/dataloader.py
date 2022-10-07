import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import csv
import random
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from PIL import Image


# Set random seed
def seed_torch(seed=1000):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_torch()


def get_dataloader(opt, train=True):
    transform = get_transform(opt, train)
    if opt.dataset == "gtsrb":
        dataset = GTSRB(opt, train, transform)
    elif opt.dataset == "mnist":
        dataset = torchvision.datasets.MNIST(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar10":
        dataset = torchvision.datasets.CIFAR10(opt.data_root, train, transform, download=True)
    elif opt.dataset == "cifar100":
        dataset = torchvision.datasets.CIFAR100(opt.data_root, train, transform, download=True)
    elif opt.dataset == "celeba":
        if train:
            split = "train"
        else:
            split = "test"
        dataset = CelebA_attr(opt, split, transform)
    elif opt.dataset == "tiny":
        dataset = Tiny(opt, train, transform)
    elif opt.dataset == "imagenet":
        if train:
            dataset = datasets.ImageFolder(os.path.join('./dataset/sub-imagenet-200', 'train'), transform)
        else:
            dataset = datasets.ImageFolder(os.path.join('./dataset/sub-imagenet-200', 'test'), transform)
    else:
        raise Exception("Invalid dataset")

    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)

    ### Train on part of the clean dataset (relates to --clean_rate) ###
    if train:
        # Train Mode
        idx = np.arange(len(dataset))
        np.random.shuffle(idx)
        training_idx = idx[:int(len(dataset) * opt.clean_rate)]
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False, sampler=SubsetRandomSampler(training_idx))
    else:
        # Test Mode
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)

    return dataloader


def get_transform(opt, train=True):
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    if train:
        if opt.dataset == 'cifar10' or opt.dataset == 'gtsrb':
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
            transforms_list.append(transforms.RandomHorizontalFlip())
        elif opt.dataset == 'cifar100':
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=4))
            transforms_list.append(transforms.RandomHorizontalFlip())
            transforms_list.append(transforms.RandomRotation(15))
        elif opt.dataset == "imagenet":
            transforms_list.append(transforms.RandomRotation(20))
            transforms_list.append(transforms.RandomHorizontalFlip(0.5))
        elif opt.dataset == "tiny":
            transforms_list.append(transforms.RandomCrop((opt.input_height, opt.input_width), padding=8))
            transforms_list.append(transforms.RandomHorizontalFlip())

    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)

class GTSRB(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(GTSRB, self).__init__()
        if train:
            self.data_folder = os.path.join(opt.data_root, "Train")
            self.images, self.labels = self._get_data_train_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)
        else:
            self.data_folder = os.path.join(opt.data_root, "Test")
            self.images, self.labels = self._get_data_test_list()
            if not os.path.isdir(self.data_folder):
                os.makedirs(self.data_folder)

        self.transforms = transforms

    def _get_data_train_list(self):
        images = []
        labels = []
        for c in range(0, 43):
            prefix = self.data_folder + "/" + format(c, "05d") + "/"
            if not os.path.isdir(prefix):
                os.makedirs(prefix)
            gtFile = open(prefix + "GT-" + format(c, "05d") + ".csv")
            gtReader = csv.reader(gtFile, delimiter=";")
            next(gtReader)
            for row in gtReader:
                images.append(prefix + row[0])
                labels.append(int(row[7]))
            gtFile.close()
        return images, labels

    def _get_data_test_list(self):
        images = []
        labels = []
        prefix = os.path.join(self.data_folder, "GT-final_test.csv")
        gtFile = open(prefix)
        gtReader = csv.reader(gtFile, delimiter=";")
        next(gtReader)
        for row in gtReader:
            images.append(self.data_folder + "/" + row[0])
            labels.append(int(row[7]))
        return images, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

class CelebA_attr(data.Dataset):
    def __init__(self, opt, split, transforms):
        self.dataset = torchvision.datasets.CelebA(root=opt.data_root, split=split, target_type="attr", download=True)
        self.list_attributes = [18, 31, 21]
        self.transforms = transforms
        self.split = split

    def _convert_attributes(self, bool_attributes):
        return (bool_attributes[0] << 2) + (bool_attributes[1] << 1) + (bool_attributes[2])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        input, target = self.dataset[index]
        input = self.transforms(input)
        target = self._convert_attributes(target[self.list_attributes])
        return (input, target)

class Tiny(data.Dataset):
    def __init__(self, opt, train, transforms):
        super(Tiny, self).__init__()
        self.opt = opt 
        self.id_dict = self._get_id_dictionary()
        if train:
            self.images, self.labels = self._get_data_train_list()
            # print(f'image shape: {(self.images).size}; lables shape: {self.labels.shape}')
        else:
            self.images, self.labels = self._get_data_test_list()
            # print(f'image shape: {(self.images).shape}; lables shape: {self.labels.shape}')
        self.transforms = transforms
    
    def _get_id_dictionary(self):
        id_dict = {}
        for i, line in enumerate(open(self.opt.data_root + '/wnids.txt', 'r')):
            line = line.split()[0]
            id_dict[line] = i
        return id_dict

    def _get_data_train_list(self):
        # print('starting loading data')
        train_data, train_labels = [], []
        for key, value in self.id_dict.items():
            train_data += [ self.opt.data_root + '/train/{}/images/{}_{}.JPEG'.format(key, key, str(i)) for i in range(500)]
            train_labels += [value]*500 
        return np.array(train_data), np.array(train_labels)

    def _get_data_test_list(self):
        test_data, test_labels = [], []
        for line in open(self.opt.data_root + '/val/val_annotations.txt'):
            img_name, class_id = line.split('\t')[:2]
            test_data.append(( self.opt.data_root + '/val/images/{}'.format(img_name)))
            test_labels.append(self.id_dict[class_id])
        return np.array(test_data), np.array(test_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = self.transforms(image)
        label = self.labels[index]
        return image, label
