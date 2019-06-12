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
import torch.nn.functional as F
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='teacher-student network')

parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./teacher-student-master/results/teacher_student_results',
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
parser.add_argument('--teacher_model_dir', metavar='TEACHER_DIR', default='./teacher-student-master/results/teacher_results/resnet_v2_1',
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
parser.add_argument('--lr', '--learning_rate', default=3e-2, type=float,
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

parser.add_argument('--teachers_inflate', default=4, type=int, metavar='TEACHERS_INFLATE', 
                    help='teacher network width inflate coefficient')

parser.add_argument('--teachers_depth', default=30, type=int, metavar='TEACHERS_DEPTH', 
                    help='teacher residual network depth (residual convolutions)')

parser.add_argument('--temperature', default=6.0, type=float, metavar='TEMPERATURE COEFFICIENT',
	                 help='temperature coefficient for kd')

parser.add_argument('--regularizer_coff', default=0.7, type=float, metavar='COEFFICIENT FOR TWO TERM',
	                 help='coefficient for two terms of kd')

parser.add_argument('--val_ratio', default=0.1, type=float, metavar='VAL_RATIO', 
                    help='validation dataset ratio')




def main():
    global args, best_prec1_val, best_prec1_test
    global best_val_epoch, best_test_epoch
    best_prec1_val = 0.
    best_prec1_test = 0.
    best_val_epoch = 0
    best_test_epoch = 0
    args = parser.parse_args()

    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
    logging.info("creating model %s", args.student_model)
    student_model = models.__dict__[args.student_model]
    student_model_config = {'input_size': args.input_size, 'dataset': args.dataset, 
                    'inflate': args.students_inflate, 'depth': args.students_depth, 'lr': args.lr, 
                    'wd': args.weight_decay}
    if args.students_model_config is not '':
        student_model_config = dict(student_model_config, **literal_eval(args.students_model_config))

    student_model = student_model(**student_model_config)
    logging.info("created model with configuration: %s", student_model_config)

    teacher_model = models.__dict__[args.teacher_model]

    # optionally resume from a checkpoint
    if args.evaluate:
        if not os.path.isfile(args.evaluate):
            parser.error('invalid checkpoint: {}'.format(args.evaluate))
        checkpoint = torch.load(args.evaluate)
        student_model.load_state_dict(checkpoint['state_dict'])
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
            student_model.load_state_dict(checkpoint['state_dict'])
            logging.info("loaded checkpoint '%s' (epoch %s)",
                         checkpoint_file, checkpoint['epoch'])
        else:
            logging.error("no checkpoint found at '%s'", args.resume)

#    num_parameters = sum([l.nelement() for l in student_model.parameters()])
#    logging.info("number of parameters: %d", num_parameters)
    

    teacher_checkpoint_file = args.teacher_model_dir
    teacher_checkpoint_file = os.path.join(teacher_checkpoint_file, 'model_best.pth.tar')
    teacher_checkpoint = torch.load(teacher_checkpoint_file)
    teacher_model_config = {'input_size': args.input_size, 'dataset': args.dataset, 
                            'inflate': args.teachers_inflate, 'depth': args.teachers_depth, 'lr': args.lr, 
                            'wd': args.weight_decay}
    teacher_model = teacher_model(**teacher_model_config)
    logging.info("created model with configuration: %s", teacher_model_config)
    teacher_model.load_state_dict(teacher_checkpoint['state_dict'])


    # Data loading code
    default_transform = {
        'train': get_transform(args.dataset,
                               input_size=args.input_size, augment=True),
        'eval': get_transform(args.dataset,
                              input_size=args.input_size, augment=False)
    }
    transform = getattr(student_model, 'input_transform', default_transform)
    regime = getattr(student_model, 'regime', {0: {'optimizer': args.optimizer,
                                           'lr': args.lr,
                                           'momentum': args.momentum,
                                           'weight_decay': args.weight_decay}})
    regime[0]['lr'] = args.lr
    regime[0]['weight_decay'] = args.weight_decay

    # define loss function (criterion) and optimizer
    criterion_train = getattr(teacher_model, 'criterion', nn.CrossEntropyLoss)()
    criterion_train.type(args.type)
    criterion_val = getattr(student_model, 'criterion', nn.CrossEntropyLoss)()
    criterion_val.type(args.type)
    student_model.type(args.type)
    teacher_model.type(args.type)

#    val_data = get_dataset(args.dataset, 'val', transform['eval'])
#    val_loader = torch.utils.data.DataLoader(
#        val_data,
#        batch_size=args.batch_size, shuffle=False,
#        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, student_model, criterion_val, 0)
        return




#    train_data = get_dataset(args.dataset, 'train', transform['train'])
#    train_loader = torch.utils.data.DataLoader(
#        train_data,
#        batch_size=args.batch_size, shuffle=True,
#        num_workers=args.workers, pin_memory=True)


    train_loader, val_loader, test_loader = create_dataloader(args.dataset,transform,args.val_ratio,args.batch_size,args.workers)
    optimizer = torch.optim.SGD(student_model.parameters(), lr=args.lr)
    logging.info('training regime: %s', regime)



    warm_up = int(0.1 * args.epochs)
    scheduler_3 = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, 
                                    gamma=0.001**(1/(args.epochs - warm_up)))


    for epoch in range(args.start_epoch, args.epochs):
        optimizer = adjust_optimizer(optimizer, epoch, regime)

        if epoch < warm_up:
            for param_group in optimizer.param_groups:
                param_group['lr'] = ((epoch + 1)/warm_up) * regime[0]['lr']
        else:
            scheduler_3.step()

#        teacher_outputs = fetch_teacher_outputs(teacher_model, train_loader)
        

        # train for one epoch
        train_loss, train_loss_teacher, train_loss_label, train_prec1, train_prec5 = train(
            train_loader, student_model, teacher_model, criterion_train, epoch, optimizer)

        # evaluate on validation set
        val_loss, val_loss_teacher, val_loss_label, val_prec1, val_prec5 = validate(
            val_loader, student_model, criterion_val, epoch)


        test_loss, test_loss_teacher, test_loss_label, test_prec1, test_prec5 = validate(
                        test_loader, student_model, criterion_val, epoch)

        # remember best prec@1 and save checkpoint
        is_best_val = val_prec1 > best_prec1_val
        if is_best_val:
            best_val_epoch = epoch + 1
        best_prec1_val = max(val_prec1, best_prec1_val)
        
        is_best_test = test_prec1 > best_prec1_test
        if is_best_test:
            best_test_epoch = epoch + 1
        best_prec1_test = max(test_prec1, best_prec1_test)

        save_checkpoint({
            'epoch': epoch + 1,
            'model': args.student_model,
            'config': student_model_config,
            'state_dict': student_model.state_dict(),
            'best_prec1_val': best_prec1_val, 
            'best_val_epoch': best_val_epoch, 
            'best_prec1_test': best_prec1_test, 
            'best_test_epoch': best_test_epoch, 
            'regime': regime
        }, is_best_val=is_best_val, is_best_test=is_best_test, path=save_path)
        logging.info('\n Epoch: {0}\t'
                     'Training Loss {train_loss:.4f} \t'
                     'Training Loss Teacher {train_loss_teacher:.4f} \t'
                     'Training Loss Label {train_loss_label:.4f} \t'
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
                             train_prec5=train_prec5, val_prec5=val_prec5, train_loss_teacher=train_loss_teacher,
                             train_loss_label=train_loss_label, test_prec1=test_prec1, test_prec5=test_prec5, 
                             test_loss=test_loss))

        results.add(epoch=epoch + 1, train_loss=train_loss, train_loss_teacher=train_loss_teacher, train_loss_label=train_loss_label,
                    val_loss=val_loss, test_loss=test_loss, train_prec1=train_prec1, val_prec1=val_prec1, test_prec1=test_prec1,
                    train_prec5=train_prec5, val_prec5=val_prec5, test_prec5=test_prec5, 
                    best_val_epoch=best_val_epoch, best_prec1_val=best_prec1_val, 
                    best_test_epoch=best_test_epoch, best_prec1_test=best_prec1_test)
        #results.plot(x='epoch', y=['train_loss', 'val_loss'],
        #             title='Loss', ylabel='loss')
        #results.plot(x='epoch', y=['train_error1', 'val_error1'],
        #             title='Error@1', ylabel='error %')
        #results.plot(x='epoch', y=['train_error5', 'val_error5'],
        #             title='Error@5', ylabel='error %')
        results.save()

# Helper function: get [batch_idx, teacher_outputs] list by running teacher model once
def fetch_teacher_outputs(teacher_model, dataloader):
    # set teacher_model to evaluation mode
    teacher_model.eval()
    teacher_outputs = []
    for i, (data_batch, labels_batch) in enumerate(dataloader):
        #print(labels_batch)
        data_batch, labels_batch = data_batch.cuda(async=True), \
                                        labels_batch.cuda(async=True)


        output_teacher_batch = teacher_model(data_batch).data.cpu().numpy()
        teacher_outputs.append(output_teacher_batch)
    #print(teacher_outputs)
    return teacher_outputs



#calculate the loss using kd
def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = args.temperature
    alpha = args.regularizer_coff
    
    """
    print(nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1))*(T*T))
    print(F.cross_entropy(outputs, labels))
    """

    KD_loss_part1 = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1), F.softmax(teacher_outputs/T, dim=1))
    KD_loss_part2 = F.cross_entropy(outputs, labels)
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)
    

    return KD_loss, KD_loss_part1, KD_loss_part2

def forward(data_loader, model, teacher_model, criterion, epoch=0, training=True, optimizer=None):
    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_part1 = AverageMeter()
    losses_part2 = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    if training:
       teacher_model.eval()
    for i, (inputs, target) in enumerate(data_loader):
        # measure data loading time
        #print(target)
        data_time.update(time.time() - end)
        if args.gpus is not None:
            target = target.cuda(async=True)
        input_var = Variable(inputs.type(args.type), volatile=not training)
        target_var = Variable(target)

        # compute output
        output = model(input_var)
        if training:
#           print(np.argmax(teacher_outputs[i],axis=1))
#           print(target)
           with torch.no_grad():
                teacher_output_batch = teacher_model(input_var)
#           print(teacher_output_batch)
           loss, loss_part1, loss_part2 = loss_fn_kd(output, target_var, teacher_output_batch)
        else:
           loss = criterion(output, target_var)
           loss_part1 = torch.zeros(1)
           loss_part2 = loss
        
        if type(output) is list:
            output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        losses_part1.update(loss_part1.item(), inputs.size(0))
        losses_part2.update(loss_part2.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

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
                         'Loss teacher {loss_part1.val:.4f} ({loss_part1.avg:.4f})\t'
                         'Loss label {loss_part2.val:.4f} ({loss_part2.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                         'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                             epoch, i, len(data_loader),
                             phase='TRAINING' if training else 'EVALUATING',
                             batch_time=batch_time,
                             data_time=data_time, loss=losses, loss_part1=losses_part1, loss_part2=losses_part2, top1=top1, top5=top5))

    return losses.avg, losses_part1.avg, losses_part2.avg, top1.avg, top5.avg


def train(data_loader, student_model, teacher_model, criterion, epoch, optimizer):
    # switch to train mode
    student_model.train()
    return forward(data_loader, student_model, teacher_model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, student_model, criterion, epoch):
    # switch to evaluate mode
    student_model.eval()
    return forward(data_loader, student_model, None, criterion, epoch,
                   training=False, optimizer=None)

def create_dataloader(name, transform, val_ratio, batch_size, workers):
    dataset_train = get_dataset(name, 'train', transform['train'])
    dataset_val = get_dataset(name, 'train', transform['eval'])
    dataset_test = get_dataset(name, 'test', transform['eval'])
    
    print(dataset_train)
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
