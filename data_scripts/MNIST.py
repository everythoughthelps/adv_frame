import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import argparse

def main(args):
    train_data = datasets.MNIST(root = args.data_root,
                                      train = True,
                                      transform = transforms.ToTensor(),
                                      target_transform = None,
                                      download = True)
    train_data_loader = torch.utils.data.DataLoader(train_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_worker)

    test_data = datasets.MNIST(root = args.data_root,
                                train = False,
                                transform=torchvision.transforms.ToTensor(),
                                target_transform = None,
                                download = True)
    test_data_loader = torch.utils.data.DataLoader(test_data,
                                                    batch_size=args.batch_size,
                                                    shuffle=False,
                                                    num_workers=args.num_worker)
    return train_data_loader, test_data_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attck template')
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--num_worker', type=int,default=4)
    args = parser.parse_args()

    main(args)

