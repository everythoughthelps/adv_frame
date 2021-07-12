import argparse
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

    return attack

def adv_img_root(data_root, method, eps):
    adv_train_save_path = os.path.join(data_root,method,str(eps),'train')
    adv_test_save_path = os.path.join(data_root,method,str(eps),'test')
    os.makedirs(adv_train_save_path,exist_ok=True)
    os.makedirs(adv_test_save_path,exist_ok=True)
    if 'CIFAR' or 'MNIST' in data_root:
        for i in range(10):
            adv_cla_train_dir = os.path.join(adv_train_save_path,str(i))
            os.makedirs(adv_cla_train_dir,exist_ok=True)
            adv_cla_test_dir = os.path.join(adv_test_save_path,str(i))
            os.makedirs(adv_cla_test_dir,exist_ok=True)
    elif 'ImageNet' in data_root:
        pass

    return adv_train_save_path, adv_test_save_path


def save_image(images,labels,save_path,iter):
    for i in range(images.size(0)):
        adv_image = images[i:i + 1, :, :, :]
        adv_image = adv_image.squeeze()
        cla = labels[i:i + 1].item()
        image_save_path = os.path.join(save_path, str(cla), str('%05d' %(iter * args.batch_size + i)) + '.jpg')
        print(image_save_path)
        adv_image = ToPILImage()(adv_image.cpu())
        adv_image.save(image_save_path)


def main(args):
    #adv_train_save_dir, adv_test_save_dir = adv_img_root(args.data_root,args.attack_method,args.eps)
    model = define_model(args.net_arch)
    model = cuda(model)
    load(model)
    attack = attack_method(args.attack_method,model)
    train_data_loader, test_data_loader = choose_data(args.dataset)

    #for i, data in enumerate(train_data_loader):
    #    images, labels = data
    #    images = images.cuda()
    #    labels = labels.cuda()
    #    adv_images = attack(images, labels)
    #    adv_images = adv_images.squeeze()
    #    save_image(adv_images,labels,adv_train_save_dir,i)

    correct = 0
    total = 0
    #with torch.no_grad():
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
        print(i / len(test_data_loader), 'test')
    acc = correct / total
    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))

        #save_image(adv_images,labels,adv_test_save_dir,i)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='attck template')
    parser.add_argument('--data_root', type=str, default='/home/yrx/data/CIFAR/')
    parser.add_argument('--dataset', type=str, default='CIFAR',choices=['CIFAR','MNIST'])
    parser.add_argument('--net_arch', type=str, default='wideresnet', choices=['resnet18', 'mnist_net', 'CIFAR_Net','wideresnet'])
    parser.add_argument('--load', type=str, default='/home/yrx/adv_frame/adv_frame/experiments/2021_06_15_14_53_36/ckp/98checkpoint.pth.tar')
    parser.add_argument('--attack_method', type=str, default='fgsm', choices=['fgsm','deepfool'])
    parser.add_argument('--eps',type=float, default=0.007)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--num_worker', type = int, default=4)

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] ='1,2,3'
    main(args)
