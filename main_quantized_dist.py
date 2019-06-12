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
from utils import *
from datetime import datetime
from ast import literal_eval
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--model', '-a', metavar='MODEL', default='resnet_preact_quan_test',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: alexnet)')

parser.add_argument('--inflate', default=1, type=int, metavar='INFLATE', 
                    help='network width inflate coefficient')

parser.add_argument('--depth', default=20, type=int, metavar='DEPTH', 
                    help='residual network depth (residual convolutions)')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results/teacher_results',
                    help='results dir')

parser.add_argument('--save', metavar='SAVE', default='resnet_preact_quan_test',
                    help='saved folder')

parser.add_argument('--dataset', metavar='DATASET', default='cifar10',
                    help='dataset name or folder')

parser.add_argument('--input_size', type=int, default=None,
                    help='image input size')

parser.add_argument('--model_config', default='',
                    help='additional architecture configuration')

if torch.cuda.is_available():
    args_type = 'torch.cuda.FloatTensor'
else:
    args_type = 'torch.FloatTensor'
parser.add_argument('--type', default=args_type,
                    help='type of tensor - e.g torch.cuda.HalfTensor')

parser.add_argument('--gpus', default='0',
                    help='gpus used for training - e.g 0,1,3')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=100, type=int,
                    metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--optimizer', default='SGD', type=str, metavar='OPTIMIZER',
                    help='optimizer function used')

parser.add_argument('--lr', '--learning_rate', default=2e-2, type=float,
                    metavar='LR', help='initial learning rate')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', type=str, metavar='FILE',
                    help='evaluate model FILE on validation set')



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
    
    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    if args.evaluate:
        validate(val_loader, model, criterion, 0)
        return    
    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)
    
    warm_up = int(0.1 * args.epochs)
        
    scheduler_3 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                    gamma=0.001**(1/(args.epochs - warm_up)))
        
    train_loss_plt = []
    train_acc_plt = []
    val_loss_plt = []
    val_acc_plt = []
    test_loss_plt = []
    test_acc_plt = []
    
    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)
        
#        if epoch < warm_up:
#            for param_group in optimizer.param_groups:
#                param_group['lr'] = ((epoch + 1)/warm_up) * regime[0]['lr']
#        else:
#            scheduler_3.step()

#        elif epoch < warm_up + cosine_epochs_1:
#            scheduler_1.step()
#        else:
#            scheduler_2.step()
        for param_group in optimizer.param_groups:
            print('learning rate: ', param_group['lr'])

        # train for one epoch
        train_loss, train_prec1, train_prec5 = train(
            train_loader, model, criterion, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_prec1, val_prec5 = validate(
            val_loader, model, criterion, epoch)
        
        
        train_loss_plt.append(train_loss)
        train_acc_plt.append(train_prec1)
        val_loss_plt.append(val_loss)
        val_acc_plt.append(val_prec1)
        

        # remember best prec@1 and save checkpoint
        is_best_val = val_prec1 > best_prec1_val
        if is_best_val:
            best_val_epoch = epoch + 1
        best_prec1_val = max(val_prec1, best_prec1_val)

        
        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.model,
            'config': model_config,
            'state_dict': model.state_dict(),
            'best_prec1_val': best_prec1_val,
            'regime': regime
        }, is_best_val, path=save_path)
                
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Prec@1 {train_prec1:.3f} \t'
                     'Training Prec@5 {train_prec5:.3f} \t'
                     'Validation Loss {val_loss:.4f} \t'
                     'Validation Prec@1 {val_prec1:.3f} \t'
                     'Validation Prec@5 {val_prec5:.3f} \t'

                     .format(epoch + 1, train_loss=train_loss, val_loss=val_loss,
                             train_prec1=train_prec1, val_prec1=val_prec1,
                             train_prec5=train_prec5, val_prec5=val_prec5))

        results.add(epoch=epoch+1, train_loss=train_loss, val_loss=val_loss, train_prec1=train_prec1, val_prec1=val_prec1, train_prec5=train_prec5, 
                    val_prec5=val_prec5, best_val_epoch=best_val_epoch, best_prec1_val=best_prec1_val, 
                    best_test_epoch=best_test_epoch, best_prec1_test=best_prec1_test)

        results.save()
    
    epoch_plt = range(len(train_loss_plt))
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(epoch_plt, train_loss_plt, label='train loss')
    plt.plot(epoch_plt, val_loss_plt, label='val loss')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss_pic.jpg'))
    plt.close()
    
    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(epoch_plt, train_acc_plt, label='train acc')
    plt.plot(epoch_plt, val_acc_plt, label='val acc')
    plt.legend()
    plt.savefig(os.path.join(save_path, 'acc_pic.jpg'))
    plt.close()
    


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
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

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        if type(output) is list:
            output = output[0]
        
        print(output[0])
        print(loss)
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], inputs.size(0))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        if training:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            for p in list(model.parameters()):
                if hasattr(p,'org'):
                    p.data.copy_(p.org)
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


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


if __name__ == '__main__':
    main()
