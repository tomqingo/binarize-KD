import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from vgg import vgg
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from torchvision.utils import save_image
import numpy as np
from torchvision import models
import sklearn
from sklearn.metrics import  
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
parser.add_argument('-b', '--batch-size', default=500, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--resume', default='./c10_vgg_float_1', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')


def main():
    global args
    args = parser.parse_args()

    
    model_config = {'input_size': args.input_size, 'dataset': args.dataset}

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None
    
    model = vgg(**model_config)
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
    print(val_data)

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

        #print(inputs.shape)
        #print(targets.shape)
    for i, (inputs,targets) in enumerate(val_loader):
        #print(inputs.shape)
        #print(targets.shape)
        category_index = category_division(targets)
        #print(category_index)
        if i == 1:
            break
        if args.gpus is not None:
           targets = targets.cuda(async=True)
           inputs = inputs.cuda(async=True)
        print(targets)
        outputs = inputs
        for index,layer in enumerate(model.features):
            #print(index)
            #print(layer)
            outputs = layer(outputs)
            print(outputs.shape)
            #outputs_statistic = category_data_hist(outputs, category_index, 10)
            #print(outputs_statistic)
            outputs_layer = outputs.data.cpu().numpy()
            outputs_layer = outputs_layer.reshape(args.batch_size, -1)
            print(outputs_layer.shape)
            outputs_reduct = dimension_reduction(outputs_layer)
            print(outputs_reduct)
            outputs_category = category_data_accumulate(outputs_reduct, category_index)
            print(outputs_category)
            save_path = './feature'+str(index)+'.png'
            category_data_visualize(save_path, outputs_category)
            #plt.figure()
            #sns.heatmap(outputs_statistic)
            #plt.savefig('./feature_'+str(index)+'.png')
        outputs = outputs.view(-1, 512*4*4)
        for index,layer in enumerate(model.classifier):
            #print(index)
            #print(layer)
            outputs = layer(outputs)
            #outputs_statistic = category_data_hist(outputs, category_index, 10)
            #print(outputs_statistic)
            outputs_layer = outputs.data.cpu().numpy()
            outputs_layer = outputs_layer.reshape(args.batch_size, -1)
            print(outputs_layer.shape)
            outputs_reduct = dimension_reduction(outputs_layer)
            print(outputs_reduct)
            outputs_category = category_data_accumulate(outputs_reduct, category_index)
            print(outputs_category)
            save_path = './classeify'+str(index)+'.png'
            category_data_visualize(save_path, outputs_category)
            #plt.figure()
            #sns.heatmap(outputs_statistic)
            #plt.savefig('./classifier_'+str(index)+'.png')
            




def category_division(targets):
    targets_new = targets.numpy()
    idx_sorted = np.argsort(targets_new)
    num_classes = targets[idx_sorted[-1]] + 1
    category_collect = []
    for category in range(num_classes):
        category_same_index = np.where(targets_new == category)
        category_collect.append(category_same_index)
    return category_collect

def dimension_reduction(layer_outputs):
    #pca = PCA(n_components=3)
    #layer_outputs_reduction = pca.fit_transform(layer_outputs)
    tsne = TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0, random_state=None, method=’barnes_hut’, angle=0.5)
    layer_outputs_reduction = tsne.fit_transform(layer_outputs)
    return layer_outputs_reduction

def category_data_accumulate(layer_outputs_reduct, category_index):
    category_data_collect = []
    for category, index in enumerate(category_index):
        category_layer_outputs = layer_outputs_reduct[index]
        category_data_collect.append(category_layer_outputs)
    return category_data_collect

def category_data_visualize(save_path, category_data_collect):
    plt.figure()
    for category_index in range(len(category_data_collect)):
        category_data = category_data_collect[category_index]
        plt.scatter(category_data[:,0], category_data[:,1], label=str(category_index))
    plt.xlabel('output1')
    plt.ylabel('output2')
    plt.legend()
    plt.savefig(save_path)

def category_data_visualize_3D(save_path, category_data_collect):
    fig = plt.figure()
    ax = Axes3D(fig)
    for category_index in range(len(category_data_collect)):
        category_data = category_data_collect[category_index]
        ax.scatter(category_data[:,0], category_data[:,1], category_data[:,2], label=str(category_index))
    ax.set_xlabel('output1')
    ax.set_ylabel('output2')
    ax.set_zlabel('output3')
    ax.legend()
    plt.savefig(save_path)
    

def category_data_hist(layer_outputs, category_index, groups):
    
    value_statistic_matrix = np.zeros((len(category_index),groups+1))
    min_value = torch.min(layer_outputs)
    max_value = torch.max(layer_outputs)
    value_gap = (max_value-min_value)/groups
    layer_outputs_new = layer_outputs.data.cpu().numpy()
    value_range = np.arange(min_value.data.cpu().numpy(),max_value.data.cpu().numpy(),value_gap.data.cpu().numpy())
    if value_range[-1]!=max_value.data.cpu().numpy():
       value_range = np.append(value_range, max_value.data.cpu().numpy())
    #print(value_range)
    #print(value_range.shape)
    for category, index in enumerate(category_index):
        category_layer_outputs = layer_outputs_new[index]
        category_layer_outputs = category_layer_outputs.reshape(-1)
        for value_index in range(value_range.shape[0]-1):
            #print(category_layer_outputs>=value_range[value_index])
            #print(category_layer_outputs<value_range[value_index+1])
            if value_index == value_range.shape[0]-2:
               value_statistic = np.sum((category_layer_outputs>=value_range[value_index]) & (category_layer_outputs<=value_range[value_index+1]))
            else:
               value_statistic = np.sum((category_layer_outputs>=value_range[value_index]) & (category_layer_outputs<value_range[value_index+1]))
            #print(value_statistic)
            value_statistic_matrix[category][value_index] = value_statistic/category_layer_outputs.shape[0]
        #print(np.sum(value_statistic_matrix,axis=1))
    return value_statistic_matrix


if __name__ == '__main__':
    main()

import argparse
