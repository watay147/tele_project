import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import models
import utils
import os
import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataselect', type=int, required=True, help='1. svhn->mnist 2. mnist->svhn 3. cifar10->stl10 4. stl10->cifar10')
    parser.add_argument('--dataroot', required=True, help='path to source dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='Number of filters to use in the generator network')
    parser.add_argument('--ndf', type=int, default=64, help='Number of filters to use in the discriminator network')
    parser.add_argument('--gpu', type=int, default=1, help='GPU to use, -1 for CPU training')
    parser.add_argument('--checkpoint_dir', default='results/models', help='folder to load model checkpoints from')
    parser.add_argument('--method', default='GTA', help='Method to evaluate| GTA, sourceonly')
    parser.add_argument('--model_best', type=int, default=0, help='Flag to specify whether to use the best validation model or last checkpoint| 1-model best, 0-current checkpoint')

    opt = parser.parse_args()
    if opt.dataselect not in [1,2,3,4]:
        exit('Please select the target and source dataset! \n 1. svhn->mnist 2. mnist->svhn 3. cifar10->stl10 4. stl10->cifar10')

    # GPU/CPU flags
    cudnn.benchmark = True
    if torch.cuda.is_available() and opt.gpu == -1:
        print("WARNING: You have a CUDA device, so you should probably run with --gpu [gpu id]")
    if opt.gpu>=0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu)

    # Creating data loaders
    if opt.dataselect in [1, 2]:
        mean = np.array([0.44, 0.44, 0.44])
        std = np.array([0.19, 0.19, 0.19])
    else:
        mean = np.array([0.4913997551666284, 0.48215855929893703, 0.4465309133731618])
        std = np.array([0.24703225141799082, 0.24348516474564, 0.26158783926049628])

    if opt.dataselect == 1:
        target_root = os.path.join(opt.dataroot, 'mnist/trainset')
    elif opt.dataselect == 2:
        target_root = os.path.join(opt.dataroot, 'svhn/trainset')
    elif opt.dataselect == 3:
        target_root = os.path.join(opt.dataroot, 'stl10/trainset')
    elif opt.dataselect == 4:
        target_root = os.path.join(opt.dataroot, 'cifar10/trainset')

    transform_target = transforms.Compose([transforms.Resize(opt.imageSize), transforms.ToTensor(), transforms.Normalize(mean,std)])
    target_test = dset.ImageFolder(root=target_root, transform=transform_target)
    targetloader = torch.utils.data.DataLoader(target_test, batch_size=opt.batchSize, shuffle=False, num_workers=2)

    nclasses = len(target_test.classes)
    
    # Creating and loading models
    
    netF = models._netF(opt)
    netC = models._netC(opt, nclasses)
    
    if opt.method == 'GTA':
        if opt.model_best == 0: 
            netF_path = os.path.join(opt.checkpoint_dir, 'netF.pth')
            netC_path = os.path.join(opt.checkpoint_dir, 'netC.pth')
        else:
            netF_path = os.path.join(opt.checkpoint_dir, 'model_best_netF.pth')
            netC_path = os.path.join(opt.checkpoint_dir, 'model_best_netC.pth')
    
    elif opt.method == 'sourceonly':
        if opt.model_best == 0: 
            netF_path = os.path.join(opt.checkpoint_dir, 'netF_sourceonly.pth')
            netC_path = os.path.join(opt.checkpoint_dir, 'netC_sourceonly.pth')
        else:
            netF_path = os.path.join(opt.checkpoint_dir, 'model_best_netF_sourceonly.pth')
            netC_path = os.path.join(opt.checkpoint_dir, 'model_best_netC_sourceonly.pth')
    else:
        raise ValueError('method argument should be sourceonly or GTA')
        
    netF.load_state_dict(torch.load(netF_path))
    netC.load_state_dict(torch.load(netC_path))
    
    if opt.gpu>=0:
        netF.cuda()
        netC.cuda()
        
    # Testing
    
    netF.eval()
    netC.eval()
        
    total = 0
    correct = 0

    for i, datas in enumerate(targetloader):
        inputs, labels = datas
        if opt.gpu>=0:
            inputs, labels = inputs.cuda(), labels.cuda()
        inputv, labelv = Variable(inputs, volatile=True), Variable(labels)

        outC = netC(netF(inputv))
        _, predicted = torch.max(outC.data, 1)        
        total += labels.size(0)
        correct += ((predicted == labels.cuda()).sum())
        
    test_acc = 100*float(correct)/total
    print('Test Accuracy: %f %%' % (test_acc))


if __name__ == '__main__':
    main()

