import argparse
import os

import time
import torch
import torchvision
import torch.nn as nn
from model import net_module



def setup_logging():
    save_path = os.path.join(args.experiment, time.strftime("%Y_%m_%d_%H_%M_%S"))
    os.makedirs(save_path)
    ckp_path = os.path.join(save_path, 'ckp')
    os.mkdir(ckp_path)
    args.ckp_path = ckp_path
    args.save_path = save_path
    with open(os.path.join(save_path, 'records_batch.csv'), 'w') as f:
        f.write('Epoch,Batch,Time,Time_sum,Loss,Loss_avg\n')

    with open(os.path.join(save_path, 'args.txt'), 'w') as f:
        f.write(str(args))

    with open(os.path.join(save_path, 'records_val.csv'), 'w') as f:
        f.write('Epoch,Loss\n')

def cuda(model):
    if torch.cuda.is_available():
        model = model.cuda()
        device_num = torch.cuda.device_count()
        print('you have %d available GPU' % (device_num))
        if device_num > 1:
            device_ids = [x for x in range(device_num)]
            model = torch.nn.DataParallel(model, device_ids=device_ids)
            args.batch_size *= device_num
    return model

def load(model):
    if args.load:
        model.load_state_dict(torch.load(args.load)['state_dict'])
        print('Model loaded from {}'.format(args.load))


def eval(model, test_data_loader):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_data_loader):
        image, label = data
        image = image.cuda()
        label = label.cuda()
        outputs = model(image)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        print(i / len(test_data_loader), 'test')
    acc = correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    with open(os.path.join(args.save_path, './records_val.csv'), 'a') as f:
        f.write('%d,%f\n' % (1, acc))


def define_model(net_arch, dataset='ImageNet'):
    if net_arch == 'resnet18':
        from model import ResNet
        if dataset == 'ImageNet':
            model = ResNet.resnet18_ImageNet
        if dataset == 'CIFAR':
            model = ResNet.resnet18_CIFAR
    elif net_arch == 'MNIST_Net':
        from model import MNIST_Net
        model = MNIST_Net.MNIST_net
    elif net_arch == 'CIFAR_Net':
        from model import CIFAR_Net
        model = CIFAR_Net.CIFAR_Net
    elif net_arch == 'wideresnet':
        from model import wideresnet
        model = wideresnet.WideResNet()

    return model

def choose_data(dataset):
    if 'MNIST' in dataset:
        from data_scripts import MNIST
        train_data_loader, test_data_loader = MNIST.main(args)
    elif 'CIFAR' in dataset:
        from data_scripts import CIFAR
        #train_data_loader, test_data_loader = CIFAR.diy_dataloader(args)
        train_data_loader, test_data_loader = CIFAR.encapsulate_loader(args)

    return train_data_loader, test_data_loader

def main(args):
    setup_logging()
    model = define_model(args.net_arch,args.dataset)
    model = cuda(model)
    train_data_loader, test_data_loader = choose_data(args.dataset)
    load(model)
    for i,data in enumerate(test_data_loader):
        img, label = data
        print(type(img))

    eval(model, test_data_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval script')
    parser.add_argument('--data_root', type=str, default='/home/yrx/data/CIFAR/origin/')
    parser.add_argument('--dataset', type=str, default='CIFAR',choices=['ImageNet','CIFAR','MNIST'])
    parser.add_argument('--net_arch', type=str, default='wideresnet', choices=['resnet18', 'mnist_net', 'CIFAR_Net','wideresnet'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--load', type=str, default='/home/yrx/adv_frame/adv_frame/experiments/2021_06_15_14_53_36/ckp/99checkpoint.pth.tar')
    parser.add_argument('--experiment', default='./experiments', type=str, help='path of experiments')

    args = parser.parse_args()
    main(args)
