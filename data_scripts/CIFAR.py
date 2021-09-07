import os

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision import datasets
from torch.utils.data import Dataset
import argparse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    torch.backends.cudnn.deterministic = True

def encapsulate_loader(args):
    setup_seed(20)
    train_data = datasets.CIFAR10(root = args.data_root,
                                      train = True,
                                      transform = transforms.ToTensor(),
                                      target_transform = None,
                                      download = True)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_worker)

    test_data = datasets.CIFAR10(root = args.data_root,
                                train = False,
                                transform=torchvision.transforms.ToTensor(),
                                target_transform = None,
                                download = True)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_worker)
    return train_data_loader, test_data_loader

class CIFAR(Dataset):
    def __init__(self,root,train,transform):
        super(CIFAR, self).__init__()
        self.transform = transform
        self.root = root
        if train:
            self.mode = 'train'
        else:
            self.mode = 'test'
        filelist = []
        for i in range(5):
            name = os.path.join(root, str('data_batch_%d' % (i + 1)))
            filelist.append(name)
        self.folder = os.path.join(self.root, self.mode)
        self.folders = os.listdir(self.folder)
        self.images = []
        for i in self.folders:
            images = os.listdir(os.path.join(self.folder,i))
            for image in images:
                img_abs = os.path.join(self.folder,str(i),str(image))
                self.images.append(img_abs)


    def __getitem__(self, item):
        image = self.images[item]
        label = int(image.split('/')[-2])
        image = Image.open(image)

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)

def diy_dataloader(args):
    setup_seed(20)
    train_data = CIFAR(args.data_root,True, transform=transforms.ToTensor())
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,
                                                    num_workers=args.num_worker)
    test_data = CIFAR(args.data_root,False, transform=transforms.ToTensor())
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                   batch_size=args.batch_size,
                                                   shuffle=True,
                                                   num_workers=args.num_worker)
    return train_data_loader, test_data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attck template')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--num_worker', type=int,default=4)
    args = parser.parse_args()


