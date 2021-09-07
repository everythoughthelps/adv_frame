import argparse
from copy import deepcopy

import torchattacks
import os

import time
import torchvision
import torch.nn as nn
import torch
from model import net_module


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

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

def attack_method(method,model):
    if method == 'fgsm':
        attack = torchattacks.FGSM(model,eps=args.eps)
    return attack

def load(model):
    if args.load:
        model.load_state_dict(torch.load(args.load)['state_dict'])
        print('Model loaded from {}'.format(args.load))

def train(model, train_data_loader, optimizer, criterion, epoch):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, data in enumerate(train_data_loader):
        image, label = data
        image = image.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        losses.update(loss.item(), image.size(0))
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        print('%d,%d/%d,%f' % (epoch, i, len(train_data_loader), loss.item()))
        with open(os.path.join(args.save_path, 'records_batch.csv'), 'a') as f:
            f.write('%d,%d/%d,%f,%f,%f,%f\n' % (
                epoch, i, len(train_data_loader), batch_time.val, batch_time.sum, losses.val, losses.avg))

def train_adv_exmp(model, train_data_loader, optimizer, criterion, dist_criterion, epoch):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for i, data in enumerate(train_data_loader):
        image, label = data
        image = image.cuda()
        label = label.cuda()
        attack = attack_method(args.attack_method, model)
        adv_image = attack(image, label)

        optimizer.zero_grad()
        output_adv, fea = model(adv_image)
        entropy_loss = criterion(output_adv, label)
        losses.update(entropy_loss.item(), image.size(0))
        entropy_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        print('%d,%d/%d,%f' % (epoch, i, len(train_data_loader), entropy_loss.item()))
        with open(os.path.join(args.save_path, 'records_adv_batch.csv'), 'a') as f:
            f.write('%d,%d/%d,%f,%f,%f,%f\n' % (
                epoch, i, len(train_data_loader), batch_time.val, batch_time.sum, losses.val, losses.avg))

def test_adv_exmp(model, test_data_loader, epoch):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_data_loader):
        image, label = data
        image = image.cuda()
        label = label.cuda()
        attack = attack_method(args.attack_method, model)
        adv_images = attack(image, label)
        outputs, fea = model(adv_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        print('%d,%d/%d,%s' % (epoch, i, len(test_data_loader), 'test process'))

    acc = correct / total
    print('Accuracy of the network on the 10000 test adversarial images: %d %%' % (
            100 * correct / total))
    with open(os.path.join(args.save_path, './records_adv_val.csv'), 'a') as f:
        f.write('%d,%f\n' % (epoch, acc))
    return acc

def test(model, test_data_loader, epoch):
    model.eval()
    correct = 0
    total = 0
    for i, data in enumerate(test_data_loader):
        image, label = data
        image = image.cuda()
        label = label.cuda()
        outputs, fea = model(image)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        print('%d,%d/%d,%s' % (epoch, i, len(test_data_loader), 'test process'))

    acc = correct / total
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    with open(os.path.join(args.save_path, './records_val.csv'), 'a') as f:
        f.write('%d,%f\n' % (epoch, acc))
    return acc

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 10:
        lr = args.lr * 0.1
    if epoch >= 15:
        lr = args.lr * 0.01
    if epoch >= 20:
        lr = args.lr * 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

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
        train_data_loader, test_data_loader = CIFAR.encapsulate_loader(args)

    return train_data_loader, test_data_loader

def main(args):
    setup_logging()
    model = define_model(args.net_arch,args.dataset)
    model = cuda(model)
    train_data_loader, test_data_loader = choose_data(args.dataset)
    load(model)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    dist_criterion = nn.CosineEmbeddingLoss(margin=0)

    for i in range(args.epoch):
        adjust_learning_rate(optimizer, i)
        #test(model, test_data_loader, i)
        #test_adv_exmp(model, test_data_loader, i)
        #train(model, train_data_loader, optimizer, criterion, i)
        train_adv_exmp(model, train_data_loader, optimizer, criterion, dist_criterion, i)
        if i == args.epoch -1:
            test(model, test_data_loader, i)
            test_adv_exmp(model, test_data_loader, i)

        save_checkpoint({'state_dict': model.state_dict()},
            filename=os.path.join(args.ckp_path, '%02dcheckpoint.pth.tar' % i))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train script')
    parser.add_argument('--attack_method', type=str, default='fgsm', choices=['fgsm','deepfool'])
    parser.add_argument('--data_root', type=str, default='/home/panmeng/data/')
    parser.add_argument('--dataset', type=str, default='CIFAR',choices=['ImageNet','CIFAR','MNIST'])
    parser.add_argument('--net_arch', type=str, default='wideresnet', choices=['resnet18', 'mnist_net', 'CIFAR_Net','wideresnet'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_worker', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--eps',type=float, default=0.03137255)
    parser.add_argument('--load', type=str, default='/home/panmeng/adv_frame/adv_frame/experiments/baseline/ckp/23checkpoint.pth.tar')
    parser.add_argument('--experiment', default='./experiments', type=str, help='path of experiments')

    args = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] ='0,1,2'
    main(args)
