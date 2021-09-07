import argparse
import numpy as np
import os
from torchvision.transforms import ToPILImage
from PIL import Image
import torch
import torchattacks


def choose_data(dataset):
    if 'MNIST' in dataset:
        from data_scripts import MNIST
        train_data_loader, test_data_loader = MNIST.main(args)
    elif 'CIFAR' in dataset:
        from data_scripts import CIFAR
        train_data_loader, test_data_loader = CIFAR.encapsulate_loader(args)

    return train_data_loader, test_data_loader

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

def load(model):
    if args.load:
        model.load_state_dict(torch.load(args.load)['state_dict'])
        print('Model loaded from {}'.format(args.load))

def attack_method(method,model):
    if method == 'fgsm':
        attack = torchattacks.FGSM(model,eps=args.eps)
    elif method == 'bim':
        attack = torchattacks.BIM(model, eps= args.eps, alpha=1/255, steps=0)
    elif method == 'deepfool':
        attack = torchattacks.DeepFool(model)
    elif method == 'pgd':
        attack = torchattacks.PGD(model,eps=args.eps,alpha=1/255)

    return attack


def main(args):
    #adv_train_save_dir, adv_test_save_dir = adv_img_root(args.data_root,args.attack_method,args.eps)
    model = define_model(args.net_arch)
    model = cuda(model)
    load(model)
    attack = attack_method(args.attack_method,model)
    train_data_loader, test_data_loader = choose_data(args.dataset)

    correct = 0
    total = 0
    #with torch.no_grad():
    dump = np.zeros((10,10))
    for i, data in enumerate(test_data_loader):
        model.eval()
        images, label = data
        images = images.cuda()
        label = label.cuda()
        adv_images = attack(images, label)
        outputs = model(adv_images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        for j in range(len(label)):
            dump[label[j]][predicted[j]] += 1
        print('%d/%d' % (i, len(test_data_loader)))
    acc = correct / total
    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))
    np.save('pgd.npy',dump)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attck template')
    parser.add_argument('--data_root', type=str, default='/home/panmeng/data/')
    parser.add_argument('--dataset', type=str, default='CIFAR',choices=['CIFAR','MNIST'])
    parser.add_argument('--net_arch', type=str, default='wideresnet', choices=['resnet18', 'mnist_net', 'CIFAR_Net','wideresnet'])
    parser.add_argument('--load', type=str, default='/home/panmeng/adv_frame/adv_frame/experiments/baseline/ckp/15checkpoint.pth.tar')
    parser.add_argument('--attack_method', type=str, default='pgd',choices=['fgsm','deepfool','bim','pgd'])
    parser.add_argument('--eps',type=float, default=8/255)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--num_worker', type = int, default=4)

    args = parser.parse_args()
    #os.environ['CUDA_VISIBLE_DEVICES'] ='1,2,3'
    main(args)
