import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from resnet_preact_quan_test import resnet_preact_quan_test
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from torchvision.utils import save_image
import numpy as np
from torchvision import models
import sklearn
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

parser = argparse.ArgumentParser(description='Visualization the result of different layer')

parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')
parser.add_argument('--type', default='torch.cuda.FloatTensor',
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='./resnet20_4_quan+resnet_float_30+0.9+6', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--inflate', default=4, type=int, metavar='INFLATE', 
                    help='network width inflate coefficient')
parser.add_argument('--depth', default=20, type=int, metavar='DEPTH', 
                    help='students residual network depth (residual convolutions)')


def main():
    global args
    args = parser.parse_args()

    
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 'inflate':args.inflate, 'depth':args.depth}

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None
    
    #model = vgg(**model_config)
    model = resnet_preact_quan_test(**model_config)
#    model = models.vgg16(pretrained=True)
    

    checkpoint_file = args.resume
    checkpoint_file = os.path.join(checkpoint_file, 'model_best_val.pth.tar')
    checkpoint = torch.load(checkpoint_file)
    #print(checkpoint['state_dict'])
    model.load_state_dict(checkpoint['state_dict'])

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    model.type(args.type)

    val_data = get_dataset(args.dataset, 'val', transform['eval'])

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

        #print(inputs.shape)
        #print(targets.shape)
    for i, (inputs,targets) in enumerate(val_loader):
        #print(inputs.shape)
        #print(targets.shape)
        #print(category_index)
        if args.gpus is not None:
           targets = targets.cuda(async=True)
           inputs = inputs.cuda(async=True)
        outputs = model(inputs)
        category_outputs = torch.max(outputs, axis=1)
        confusion_matrix_results = confusion_matrix(category_outputs.cpu().numpy(), outputs.cpu().numpy())
        if i != 0:
            confusion_matrix_results = confusion_matrix_results + confusion_tmp
        confusion_matrix_tmp = confusion_matrix_results

    print(confusion_matrix_results)
    np.savetxt(os.path.join(args.resume, 'confusion_test.csv'))
    plt.figure()
    sns.heatmap(confusion_matrix_results, annot=True)
    plt.savefig(os.path.join(args.resume, 'confusion_test.png'))