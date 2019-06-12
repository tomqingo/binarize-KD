# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 21:13:41 2018

@author: Chen
"""

import argparse
import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from my_utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import math


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='teacher-student network')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results/teacher_student_results',
                    help='results dir')
parser.add_argument('--save', metavar='SAVE', default='',
                    help='saved folder')
parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')
parser.add_argument('--teacher_model', '-a', metavar='MODEL_TEACHER', default='resnet',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')
parser.add_argument('--teacher_model_dir', metavar='TEACHER_DIR', default='./results/teacher_results/resnet_cifar10_inflate4',
                     help='teacher dir to load model')
parser.add_argument('--student_model', metavar='MODEL_STUDENT', default='resnet_preact_quan_test',
                     choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet)')
parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')
parser.add_argument('--students_model_config', default='',
                    help='additional student architecture configuration')
if torch.cuda.is_available():
    args_type = 'torch.cuda.FloatTensor'
else:
    args_type = 'torch.FloatTensor'
parser.add_argument('--type', default=args_type,
                    help='type of tensor - e.g torch.cuda.HalfTensor')
parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPT',
                    help='optimizer function used')
parser.add_argument('--lr', '--learning_rate', default=2e-2, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')
parser.add_argument('--students_inflate', default=1, type=int, metavar='STUDENTS_INFLATE', 
                    help='student network width inflate coefficient')

parser.add_argument('--students_depth', default=20, type=int, metavar='STUDENTS_DEPTH', 
                    help='students residual network depth (residual convolutions)')

parser.add_argument('--teachers_inflate', default=1, type=int, metavar='TEACHERS_INFLATE', 
                    help='teacher network width inflate coefficient')

parser.add_argument('--teachers_depth', default=20, type=int, metavar='TEACHERS_DEPTH', 
                    help='teacher residual network depth (residual convolutions)')

parser.add_argument('--temperature', default=3.0, type=float, metavar='TEMPERATURE COEFFICIENT',
                     help='temperature coefficient for kd')

parser.add_argument('--regularizer_coff', default=0.7, type=float, metavar='COEFFICIENT FOR TWO TERM',
                     help='coefficient for two terms of kd')

def main():
    
    global args, best_prec1_val, best_prec1_test
    global best_val_epoch, best_test_epoch
    
    best_prec1_val = 0.
    best_prec1_test = 0.
    best_val_epoch = 0
    best_test_epoch = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/eval_tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        save_path = os.path.join(args.results_dir, args.save)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.path.join(args.results_dir, args.save)
        folder_idx = 0
        while os.path.exists(save_path + '_' + str(folder_idx)):
            folder_idx += 1
        save_path = save_path + '_' + str(folder_idx)
        os.makedirs(save_path)
        
#    save_path = os.path.join(args.results_dir, args.save)
#    if not os.path.exists(save_path):
#        os.makedirs(save_path)

    setup_logging(os.path.join(save_path, 'log.txt'))
    results_file = os.path.join(save_path, 'results.%s')
    results = ResultsLog(results_file % 'csv', results_file % 'html')

    logging.info("saving to %s", save_path)
    logging.debug("run arguments: %s", args)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # create model
    logging.info("creating model %s", args.model)
    model = models.__dict__[args.model]
    model_config = {'input_size': args.input_size, 'dataset': args.dataset, 
                    'inflate': args.inflate, 'depth': args.depth, 'lr': args.lr, 
                    'wd': args.weight_decay}

    if args.model_config is not '':
        model_config = dict(model_config, **literal_eval(args.model_config))

    model = model(**model_config)
    logging.info("created model with configuration: %s", model_config)

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint '%s' (epoch %s)",
                     args.evaluate, checkpoint['epoch'])
    elif args.resume:
        checkpoint_file = args.resume
        if os.path.isdir(checkpoint_file):
            results.load(os.path.join(checkpoint_file, 'results.csv'))
            checkpoint_file = os.path.join(
                checkpoint_file, 'model_best.pth.tar')
        if os.path.isfile(checkpoint_file):
            logging.info("loading checkpoint '%s'", args.resume)
            checkpoint = torch.load(checkpoint_file)
            args.start_epoch = checkpoint['epoch'] - 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(model, 'input_transform', default_transform)
    regime = getattr(model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    regime[0]['weight_decay'] = args.weight_decay
    regime[0]['lr'] = args.lr
    
    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)
#    model.type(torch.FloatTensor)
    
    train_loader, val_loader, test_loader = create_dataloader(args.dataset, 
                                                              transform, 
                                                              args.val_ratio, 
                                                              args.batch_size, 
                                                              args.workers)
    
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        validate(test_loader, model, criterion, 0)
        return    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)
    
    warm_up = int(0.1 * args.epochs)
    cosine_epochs_2 = 20
    cosine_epochs_1 = args.epochs - warm_up - cosine_epochs_2
    
    scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                    T_max=cosine_epochs_1)
    scheduler_2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, 
                                    T_max=cosine_epochs_2)
    for i in range(len(scheduler_2.base_lrs)):
        scheduler_2.base_lrs[i] *= 0.5
        
#    scheduler_3 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
#                                   gamma=0.001**(1/(args.epochs - warm_up)))
    scheduler_3 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)
        
    train_loss_plt = []
    train_acc_plt = []
    val_loss_plt = []
    val_acc_plt = []
    test_loss_plt = []
    test_acc_plt = []

    params_name_collect = []
    params = model.state_dict()
    for params_name, params_value in params.items():
        if 'weight' in params_name or 'bias' in params_name:
            params_name_collect.append(params_name)

    bit_collect = []
    for m_name, m_value in list(model.modules())[0].__dict__['_modules'].items(): 
        if 'layer' in m_name:
            for sub_m_id, sub_m in enumerate(list(m_value.modules())[0:3]):
                for submm_name, submm_value in list(m_value[sub_m_id].modules())[0].__dict__['_modules'].items():
                    if 'downsample' in submm_name:
                        bit_collect.append(submm_value[0].w_bit)
                    elif 'conv' in submm_name:
                        bit_collect.append(submm_value.w_bit)
                    else:
                        continue
        if hasattr(m_value,'w_bit'):
            bit_collect.append(m_value.w_bit)
    print(bit_collect)
    
    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        
        if epoch < warm_up:
            for param_group in optimizer.param_groups:
                param_group['lr'] = ((epoch + 1)/warm_up) * regime[0]['lr']
        else:
            scheduler_3.step()
#        elif epoch < warm_up + cosine_epochs_1:
#            scheduler_1.step()
#        else:
#            scheduler_2.step()
        for param_group in optimizer.param_groups:
            print('learning rate: ', param_group['lr'])

        if epoch <=5:
           wd = 5e-4
        else:
           wd = 5e-4

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, wd, params_name_collect, bit_collect, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, wd, params_name_collect, bit_collect, epoch)
        
        test_loss, test_prec1, test_prec5 = validate(test_loader, model, 
                                                     criterion, wd, params_name_collect, bit_collect, epoch)
        
        train_loss_plt.append(train_loss)
        train_acc_plt.append(train_prec1)
        val_loss_plt.append(val_loss)
        val_acc_plt.append(val_prec1)
        test_loss_plt.append(test_loss)
        test_acc_plt.append(test_prec1)
        

        # remember best prec@1 and save checkpoint
        is_best_val = val_prec1 > best_prec1_val
        if is_best_val:
            best_val_epoch = epoch + 1
        best_prec1_val = max(val_prec1, best_prec1_val)
        
        is_best_test = test_prec1 > best_prec1_test
        if is_best_test:
            best_test_epoch = epoch + 1
        best_prec1_test = max(test_prec1, best_prec1_test)
        
        save_checkpoint({'epoch': epoch+1, 'model': args.model, 
                         'config': args.model_config, 
                         'best_prec1_val': best_prec1_val, 
                         'best_val_epoch': best_val_epoch, 
                         'best_prec1_test': best_prec1_test, 
                         'best_test_epoch': best_test_epoch, 
                         'regime': regime}, is_best_val=is_best_val, 
                         is_best_test=is_best_test, path=save_path)
                
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \t'
                     'Test Loss {test_loss:.4f} \t'
                     'Test Prec@1 {test_prec1: .3f} \t'
                     'Test Prec@5 {test_prec5: .3f} \n'
                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5, 
                             test_prec1=test_prec1, test_prec5=test_prec5, 
                             test_loss=test_loss))

        results.add(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss, 
                    test_loss=test_loss, train_prec1=train_prec1, val_prec1=val_prec1, 
                    test_prec1=test_prec1, train_prec5=train_prec5, 
                    val_prec5=val_prec5, test_prec5=test_prec5, 
                    best_val_epoch=best_val_epoch, best_prec1_val=best_prec1_val, 
                    best_test_epoch=best_test_epoch, best_prec1_test=best_prec1_test)

        results.save()
    
    epoch_plt = range(len(train_loss_plt))
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch_plt, train_loss_plt, label='train loss')
    plt.plot(epoch_plt, val_loss_plt, label='val loss')
    plt.plot(epoch_plt, test_loss_plt, label='test loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_pic.jpg'))
    plt.close()
    
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(epoch_plt, train_acc_plt, label='train acc')
    plt.plot(epoch_plt, val_acc_plt, label='val acc')
    plt.plot(epoch_plt, test_acc_plt, label='test acc')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc_pic.jpg'))
    plt.close()
    
        #model parameters distribution
    for p_id, p in enumerate(list(model.parameters())):
        if 'weight' in params_name_collect[p_id]:
            if hasattr(p,'org'):
               plt.figure()
               plt.hist(p.org.reshape(-1))
               plt.savefig(os.path.join(save_path, params_name_collect[p_id]+'.png'))
               plt.close()
    


def forward(data_loader, model, criterion, weight_decay, params_name_collect, bit_collect, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)
        print(inputs.shape)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            count = 0
            for p_id, p in enumerate(list(model.parameters())):
                print(p.data.shape)
                if hasattr(p,'org'):
                    p.data.copy_(p.org)

                    if 'weight' in params_name_collect[p_id] and bit_collect[count] > 1:
#                        print(params_name_collect[p_id])
#                        print(bit_collect[count])
#                        print(p.org)
#                        print(p.grad)
                        if bit_collect[count] == 8:
                            coeff = 100
                        elif bit_collect[count] == 4:
                            coeff = 10
                        else:
                            coeff = 1
                            
                        n = 2**(bit_collect[count]-1)-1
                        left = -1/n
                        right = 1/n
                        regularizer = torch.where((p.data>=left/2)&(p.data<=right/2), torch.sin(math.pi*n*p.data), torch.zeros_like(p.data))
                        regularizer = torch.where((p.data<left/2)&(p.data>=left), -torch.sin(math.pi*n*p.data), regularizer)
                        regularizer = torch.where((p.data>right/2)&(p.data<=right), -torch.sin(math.pi*n*p.data), regularizer)
                        for ii in range(3,n+2,2):
                            left = -ii/n
                            right = (-ii+2)/n
                            regularizer = torch.where((p.data>=left+(1/(n*2)))&(p.data<right-(1/(n*2))), torch.sin(math.pi*n*p.data), regularizer)
                            regularizer = torch.where((p.data>=left)&(p.data<left+(1/(n*2))), -torch.sin(math.pi*n*p.data), regularizer)
                            regularizer = torch.where((p.data>=right-(1/(n*2)))&(p.data<right), -torch.sin(math.pi*n*p.data), regularizer)
                        

                        for ii in range(3,n+2,2):
                            left = ii/n
                            right = (ii+2)/n
                            regularizer = torch.where((p.data>left+(1/(n*2)))&(p.data<=right-(1/(n*2))), torch.sin(math.pi*n*p.data), regularizer)
                            regularizer = torch.where((p.data>left)&(p.data<=left+(1/(n*2))), -torch.sin(math.pi*n*p.data), regularizer)
                            regularizer = torch.where((p.data>right-(1/(n*2)))&(p.data<=right), -torch.sin(math.pi*n*p.data), regularizer)
                        
#                        print(n)
#                        print(p.grad)
                        if p_id == 0:
                           print(p.grad)
                        p.grad = p.grad + regularizer*weight_decay*coeff


#                        p.grad = p.grad + 2*torch.sin(n*math.pi*p.org)*torch.cos(n*math.pi*p.org)*math.pi*weight_decay
                        count = count + 1;
                        if p_id == 0:
                           print(p.grad)
#                        print(p.grad)
            count = 0
            for p_id, p in enumerate(list(model.parameters())):
               if 'weight' in params_name_collect[p_id] and bit_collect[count] > 1:
                   print(p.grad)
               
#                    print(p[0])
            optimizer.step()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.org.copy_(p.data.clamp_(-1,1))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('{phase} - Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, weight_decay, params_name_collect, bit_collect, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, weight_decay, params_name_collect, bit_collect, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, weight_decay, params_name_collect, bit_collect, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, weight_decay, params_name_collect, bit_collect, epoch,
                   training=False, optimizer=None)


def create_dataloader(name, transform, val_ratio, batch_size, workers):
    dataset_train = get_dataset(name, 'train', transform['train'])
    dataset_val = get_dataset(name, 'train', transform['eval'])
    dataset_test = get_dataset(name, 'test', transform['eval'])
    
    idx_sorted = np.argsort(dataset_train.train_labels)
    num_classes = dataset_train.train_labels[ idx_sorted[-1] ] + 1
    samples_per_class = len(dataset_train) // num_classes
    val_len = int(val_ratio * samples_per_class)
    val_idx = np.array([], dtype=np.int32)
    train_idx = np.array([], dtype=np.int32)
    for i in range(num_classes):
        perm = np.random.permutation(range(samples_per_class))
        
        val_part = samples_per_class * i + perm[0:val_len]
        val_part = idx_sorted[val_part]
        val_idx = np.concatenate((val_idx, val_part))
        
        train_part = samples_per_class * i + perm[val_len:]
        train_part = idx_sorted[train_part]
        train_idx = np.concatenate((train_idx, train_part))
        
    sampler_train = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(val_idx)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, 
                                               shuffle=False, sampler=sampler_train, 
                                               num_workers=workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, 
                                             shuffle=False, sampler=sampler_val, 
                                             num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, 
                                              shuffle=False, num_workers=workers, 
                                              pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    main()
