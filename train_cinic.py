from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as dst
from torch.utils.data import Dataset
import argparse
import os
import time

from util import AverageMeter, accuracy, transform_time
from util import load_pretrained_model, save_checkpoint
from network import define_tsnet
from torch.autograd import Variable
from torch import Tensor
parser = argparse.ArgumentParser(description='soft target')

# various path
parser.add_argument('--save_root', type=str, default='./', help='models and logs are saved here')
parser.add_argument('--img_root', type=str, default='./CINIC-10', help='path name of image dataset')
parser.add_argument('--s_init', type=str, required=True, help='initial parameters of student model')
parser.add_argument('--t_model', type=str, required=True, help='path name of teacher model')

# training hyper parameters
parser.add_argument('--print_freq', type=int, default=10, help='frequency of showing training results on console')
parser.add_argument('--epochs', type=int, default=300, help='number of total epochs to run')
parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--num_class', type=int, default=10, help='number of classes')
parser.add_argument('--cuda', type=int, default=1)

# net and dataset choosen
parser.add_argument('--data_name', type=str, required=True, help='name of dataset')# cifar10/cifar100
parser.add_argument('--t_name', type=str, required=True, help='name of teacher')
parser.add_argument('--s_name', type=str, required=True, help='name of student')

# hyperparameter lambda
parser.add_argument('--lambda_st', type=float, default=0.1)
parser.add_argument('--T', type=float, default=3.0)

def main():
    global args
    args = parser.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.save_root,'checkpoint')):
        os.makedirs(os.path.join(args.save_root,'checkpoint'))

    if args.cuda:
        cudnn.benchmark = True

    print('----------- Network Initialization --------------')
    snet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
    # checkpoint = torch.load(args.s_init)
    # load_pretrained_model(snet, checkpoint['net'])

    tnet = define_tsnet(name=args.t_name, num_class=args.num_class, cuda=args.cuda)
    checkpoint = torch.load(args.t_model)
    load_pretrained_model(tnet, checkpoint['net'])
    tnet.eval()
    for param in tnet.parameters():
        param.requires_grad = False

    ssnet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
    checkpoint = torch.load(args.s_init)
    load_pretrained_model(ssnet, checkpoint['net'])
    ssnet.eval()
    for param in tnet.parameters():
        param.requires_grad = False

    print('-----------------------------------------------')

    # initialize optimizer
    optimizer = torch.optim.SGD(snet.parameters(),
                                lr = args.lr,
                                momentum = args.momentum,
                                weight_decay = args.weight_decay,
                                nesterov = True)

    # define loss functions
    if args.cuda:
        criterionCls = torch.nn.CrossEntropyLoss().cuda()
        criterionST  = torch.nn.KLDivLoss(reduction='sum').cuda()

    else:
        criterionCls = torch.nn.CrossEntropyLoss()
        criterionST  = torch.nn.KLDivLoss(reduction='sum')


    # define transforms
    if args.data_name == 'cifar10':
        dataset = dst.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
    elif args.data_name == 'cifar100':
        dataset = dst.CIFAR100
        mean = (0.5071, 0.4865, 0.4409)
        std  = (0.2673, 0.2564, 0.2762)
    elif args.data_name == 'CINIC10':
        dataset = dst.ImageFolder
        mean = (0.47889522, 0.47227842, 0.43047404)
        std = (0.24205776, 0.23828046, 0.25874835)
    else:
        raise Exception('invalid dataset name...')

    train_transform = transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])
    test_transform = transforms.Compose([
            # transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean,std=std)
        ])

    # define data loader
    if args.data_name == 'CINIC10':
        train_set = dataset(root=args.img_root + '/train', transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                                   pin_memory=True)
        test_set = dataset(root=args.img_root + '/test', transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                                  pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            dataset(root      = args.img_root,
                    transform = test_transform,
                    train     = True,
                    download  = True),
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            dataset(root      = args.img_root,
                    transform = test_transform,
                    train     = False,
                    download  = True),
            batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # train_dl_sort01, train_dl_sort02, train_dl_sort03, train_dl_sort04, train_dl_sort05 = sort_training_data(ssnet, train_loader, test_loader, args)

    nets = {'snet': snet, 'tnet': tnet}
    criterions = {'criterionCls': criterionCls, 'criterionST': criterionST}

    for epoch in range(1, args.epochs+1):

        num = (args.epochs) / 30
        print (num)

        epoch_start_time = time.time()

        adjust_lr(optimizer, epoch)

        if epoch <= num * 4:
            # train one epoch
            # nets = {'snet':snet, 'tnet':tnet}
            train(train_loader, nets, optimizer, criterions, epoch)
            epoch_time = time.time() - epoch_start_time
            print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

        elif num * 4 < epoch <= num * 7:

            # temp student net
            tempsnet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
            checkpoint = torch.load('./checkpoint/st_r110_r20_040.ckp')
            load_pretrained_model(tempsnet, checkpoint['snet'])
            tempsnet.eval()
            for param in tempsnet.parameters():
                param.requires_grad = False

            train_dl_sort01, train_dl_sort02, train_dl_sort03, train_dl_sort04, train_dl_sort05 = sort_training_data(tempsnet, train_loader, test_loader, args)

            traintemp(train_dl_sort01, nets, tempsnet, optimizer, criterions, epoch, args.batch_size)
            epoch_time = time.time() - epoch_start_time
            print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

        elif num * 7 < epoch <= num * 11:
            # nets = {'snet': snet, 'tnet': tnet}
            # criterions = {'criterionCls': criterionCls, 'criterionST': criterionST}

            # temp student net
            tempsnet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
            checkpoint = torch.load('./checkpoint/st_r110_r20_070.ckp')
            load_pretrained_model(tempsnet, checkpoint['snet'])
            tempsnet.eval()
            for param in tempsnet.parameters():
                param.requires_grad = False

            train_dl_sort01, train_dl_sort02, train_dl_sort03, train_dl_sort04, train_dl_sort05 = sort_training_data(
                tempsnet, train_loader, test_loader, args)

            traintemp(train_dl_sort02, nets, tempsnet, optimizer, criterions, epoch, args.batch_size)
            epoch_time = time.time() - epoch_start_time
            print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

        elif num * 11 < epoch <= num * 15:
            # nets = {'snet': snet, 'tnet': tnet}
            # criterions = {'criterionCls': criterionCls, 'criterionST': criterionST}

            # temp student net
            tempsnet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
            checkpoint = torch.load('./checkpoint/st_r110_r20_110.ckp')
            load_pretrained_model(tempsnet, checkpoint['snet'])
            tempsnet.eval()
            for param in tempsnet.parameters():
                param.requires_grad = False

            train_dl_sort01, train_dl_sort02, train_dl_sort03, train_dl_sort04, train_dl_sort05 = sort_training_data(
                tempsnet, train_loader, test_loader, args)

            traintemp(train_dl_sort03, nets, tempsnet, optimizer, criterions, epoch, args.batch_size)
            epoch_time = time.time() - epoch_start_time
            print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

        elif num * 15 < epoch <= num * 20:
            # nets = {'snet': snet, 'tnet': tnet}
            # criterions = {'criterionCls': criterionCls, 'criterionST': criterionST}

            # temp student net
            tempsnet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
            checkpoint = torch.load('./checkpoint/st_r110_r20_150.ckp')
            load_pretrained_model(tempsnet, checkpoint['snet'])
            tempsnet.eval()
            for param in tempsnet.parameters():
                param.requires_grad = False
            train_dl_sort01, train_dl_sort02, train_dl_sort03, train_dl_sort04, train_dl_sort05 = sort_training_data(
                tempsnet, train_loader, test_loader, args)
            traintemp(train_dl_sort04, nets, tempsnet, optimizer, criterions, epoch, args.batch_size)
            epoch_time = time.time() - epoch_start_time
            print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

        elif num * 20 < epoch <= num * 30:
            # nets = {'snet': snet, 'tnet': tnet}
            # criterions = {'criterionCls': criterionCls, 'criterionST': criterionST}

            # temp student net
            tempsnet = define_tsnet(name=args.s_name, num_class=args.num_class, cuda=args.cuda)
            checkpoint = torch.load('./checkpoint/st_r110_r20_200.ckp')
            load_pretrained_model(tempsnet, checkpoint['snet'])
            tempsnet.eval()
            for param in tempsnet.parameters():
                param.requires_grad = False
            train_dl_sort01, train_dl_sort02, train_dl_sort03, train_dl_sort04, train_dl_sort05 = sort_training_data(
                tempsnet, train_loader, test_loader, args)
            traintemp(train_dl_sort05, nets, tempsnet, optimizer, criterions, epoch, args.batch_size)
            epoch_time = time.time() - epoch_start_time
            print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))

        # evaluate on testing set
        print('testing the models......')
        test_start_time = time.time()
        test(test_loader, nets, criterions)
        test_time = time.time() - test_start_time
        print('testing time is {:02}h{:02}m{:02}s'.format(*transform_time(test_time)))

        # save model
        print('saving models......')
        save_name = 'st_r{}_r{}_{:>03}.ckp'.format(args.t_name[6:], args.s_name[6:], epoch)
        save_name = os.path.join(args.save_root, 'checkpoint', save_name)
        if epoch == 1:
            save_checkpoint({
                'epoch': epoch,
                'snet': snet.state_dict(),
                'tnet': tnet.state_dict(),
            }, save_name)
        else:
            save_checkpoint({
                'epoch': epoch,
                'snet': snet.state_dict(),
            }, save_name)

class MyDataset(Dataset):
    def __init__(self, imgs, labels, transform):
        self.imags = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, label = self.imags[index], self.labels[index]

        img = transforms.ToPILImage()(denormalize(img.cpu()))
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imags)

def denormalize(tensor):
    tensor = tensor * Tensor([0.24205776, 0.23828046, 0.25874835])[:, None, None]
    tensor = tensor + Tensor([0.47889522, 0.47227842, 0.43047404])[:, None, None]
    return tensor

def supp_idxs(t, value):
    return t.eq(value).nonzero().squeeze(1)

def sort_training_data(teacher_model, train_dataloader, val_dataloader, params):

    mean = (0.47889522, 0.47227842, 0.43047404)
    std = (0.24205776, 0.23828046, 0.25874835)

    train_transform = transforms.Compose([
        transforms.Pad(4, padding_mode='reflect'),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    train_images_01 = []
    train_labels_01 = []
    train_images_02 = []
    train_labels_02 = []
    train_images_03 = []
    train_labels_03 = []
    train_images_04 = []
    train_labels_04 = []
    train_images_05 = []
    train_labels_05 = []
    scores = [[], [],[],[],[],[],[],[],[],[]]
    imgs = [[], [],[],[],[],[],[],[],[],[]]
    labels = [[], [],[],[],[],[],[],[],[],[]]
    for i, (train_batch, labels_batch) in enumerate(train_dataloader):

        teacher_model.eval()
        #if params.cuda:
        train_batch, labels_batch = train_batch.cuda(), labels_batch.cuda()
        # convert to torch Variables
        train_batch, labels_batch = Variable(train_batch), Variable(labels_batch)

        # compute model output, fetch teacher output, and compute KD loss
        _, _, _, _, output_batch = teacher_model(train_batch)

        output_batch = output_batch.data.cpu()

        score, cls = torch.max(output_batch, dim=1)

        for i in range(10):
            index = supp_idxs(labels_batch, i)
            scores[i].append(score[index])
            imgs[i].append(train_batch[index])

    for i in range(10):
        print (len(scores[i]))
        scores[i] = torch.cat(scores[i])
        imgs[i] = torch.cat(imgs[i])
        scores[i], inx = torch.sort(scores[i], descending=True)
        imgs[i] = imgs[i][inx]
        print (scores[i].size())
        print (imgs[i].size())

    for i in range(10):
        train_images_01.append(imgs[i][0:1800])
        train_images_02.append(imgs[i][1800:3600])
        train_images_03.append(imgs[i][3600:5400])
        train_images_04.append(imgs[i][5400:7200])
        train_images_05.append(imgs[i][7200:9000])

    train_images_02 = train_images_01 + train_images_02
    train_images_03 = train_images_02 + train_images_03
    train_images_04 = train_images_03 + train_images_04
    train_images_05 = train_images_04 + train_images_05

    train_images_01 = torch.cat(train_images_01)
    print (train_images_01.size())
    train_images_02 = torch.cat(train_images_02)
    print (train_images_02.size())
    train_images_03 = torch.cat(train_images_03)
    print (train_images_03.size())
    train_images_04 = torch.cat(train_images_04)
    print (train_images_04.size())
    train_images_05 = torch.cat(train_images_05)
    print (train_images_05.size())

    for i in range(10):
        train_labels_01 += [i for j in range(1800)]
    print (len(train_labels_01))

    train_labels_02 = train_labels_01 + train_labels_01
    print(len(train_labels_02))
    train_labels_03 = train_labels_02 + train_labels_01
    train_labels_04 = train_labels_03 + train_labels_01
    train_labels_05 = train_labels_04 + train_labels_01

    dataset = MyDataset(train_images_01, train_labels_01, train_transform)
    train_dl_01 = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)

    dataset2 = MyDataset(train_images_02, train_labels_02, train_transform)
    train_dl_02 = torch.utils.data.DataLoader(dataset2, batch_size=128, shuffle=True)

    dataset3 = MyDataset(train_images_03, train_labels_03, train_transform)
    train_dl_03 = torch.utils.data.DataLoader(dataset3, batch_size=128, shuffle=True)

    dataset4 = MyDataset(train_images_04, train_labels_04, train_transform)
    train_dl_04 = torch.utils.data.DataLoader(dataset4, batch_size=128, shuffle=True)

    dataset5 = MyDataset(train_images_05, train_labels_05, train_transform)
    train_dl_05 = torch.utils.data.DataLoader(dataset5, batch_size=128, shuffle=True)

    return train_dl_01, train_dl_02, train_dl_03, train_dl_04, train_dl_05

def train(train_loader, nets, optimizer, criterions, epoch):

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    cls_losses = AverageMeter()
    st_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionST  = criterions['criterionST']


    snet.train()

    end = time.time()
    log_list = []
    for idx, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda()
            target = target.cuda()

        _, _, _, _, output_s = snet(img)
        _, _, _, _, output_t = tnet(img)

        cls_loss = criterionCls(output_s, target)
        st_loss  = criterionST(F.log_softmax(output_s/args.T, dim=1),
                               F.softmax(output_t/args.T, dim=1)) * (args.T*args.T) / img.size(0)
        st_loss  = st_loss * args.lambda_st
        loss = cls_loss + st_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        st_losses.update(st_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'Time:{batch_time.val:.4f} '
                  'Data:{data_time.val:.4f}  '
                  'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                  'ST:{st_losses.val:.4f}({st_losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                  epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
                  cls_losses=cls_losses, st_losses=st_losses, top1=top1, top5=top5))
            log_list.append('Epoch[{0}]:[{1:03}/{2:03}],'
                            'Time:{batch_time.val:.4f},'
                            'Data:{data_time.val:.4f},'
                            'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f}),'
                            'ST:{st_losses.val:.4f}({st_losses.avg:.4f}),'
                            'prec@1:{top1.val:.2f}({top1.avg:.2f}),'
                            'prec@5:{top5.val:.2f}({top5.avg:.2f})\n'.format(
                epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
                cls_losses=cls_losses, st_losses=st_losses, top1=top1, top5=top5))
    with open('./log_train_cinic.txt', 'a+') as f:
        f.writelines(log_list)

def traintemp(train_loader, nets, tempnets, optimizer, criterions, epoch, batchsize):

    batch_time = AverageMeter()
    data_time  = AverageMeter()
    cls_losses = AverageMeter()
    st_losses  = AverageMeter()
    temp_losses= AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionST  = criterions['criterionST']
    criterionLogits = torch.nn.MSELoss().cuda()
    kldivergence = torch.nn.KLDivLoss().cuda()

    snet.train()

    end = time.time()
    log_list = []
    for idx, (img, target) in enumerate(train_loader, start=1):
        data_time.update(time.time() - end)

        if args.cuda:
            img = img.cuda()
            target = target.cuda()

        _, _, _, _, output_s = snet(img)
        _, _, _, _, output_t = tnet(img)

        _, _, _, _, output_temp = tempnets(img)

        cls_loss = criterionCls(output_s, target)
        st_loss  = criterionST(F.log_softmax(output_s/args.T, dim=1),
                               F.softmax(output_t/args.T, dim=1)) * (args.T*args.T) / img.size(0)

        # temp_loss = kldivergence(output_temp.detach(), output_s)
        temp_loss = criterionLogits(output_s, output_temp.detach())
        #temp_loss = ((output_temp - output_s) * (output_temp - output_s)).sum() / batchsize

        # temp_loss = criterionST(F.log_softmax(output_s/args.T, dim=1),
        #                        F.softmax(output_temp/args.T, dim=1)) * (args.T*args.T) / img.size(0)

        st_loss  = st_loss * args.lambda_st
        loss = cls_loss + st_loss + 0.05*temp_loss

        prec1, prec5 = accuracy(output_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        st_losses.update(st_loss.item(), img.size(0))
        temp_losses.update(temp_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            print('Epoch[{0}]:[{1:03}/{2:03}] '
                  'Time:{batch_time.val:.4f} '
                  'Data:{data_time.val:.4f}  '
                  'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f})  '
                  'ST:{st_losses.val:.4f}({st_losses.avg:.4f})  '
                  'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                  'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(
                  epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
                  cls_losses=cls_losses, st_losses=st_losses, top1=top1, top5=top5))
            log_list.append('Epoch[{0}]:[{1:03}/{2:03}],'
                            'Time:{batch_time.val:.4f},'
                            'Data:{data_time.val:.4f},'
                            'Cls:{cls_losses.val:.4f}({cls_losses.avg:.4f}),'
                            'ST:{st_losses.val:.4f}({st_losses.avg:.4f}),'
                            'prec@1:{top1.val:.2f}({top1.avg:.2f}),'
                            'prec@5:{top5.val:.2f}({top5.avg:.2f})\n'.format(
                epoch, idx, len(train_loader), batch_time=batch_time, data_time=data_time,
                cls_losses=cls_losses, st_losses=st_losses, top1=top1, top5=top5))
    with open('./log_train_cinic.txt', 'a+') as f:
        f.writelines(log_list)

def test(test_loader, nets, criterions):
    cls_losses = AverageMeter()
    st_losses  = AverageMeter()
    top1       = AverageMeter()
    top5       = AverageMeter()

    snet = nets['snet']
    tnet = nets['tnet']

    criterionCls = criterions['criterionCls']
    criterionST  = criterions['criterionST']

    snet.eval()

    end = time.time()
    log_list = []
    for idx, (img, target) in enumerate(test_loader, start=1):
        if args.cuda:
            img = img.cuda()
            target = target.cuda()

        with torch.no_grad():
            _, _, _, _, output_s = snet(img)
            _, _, _, _, output_t = tnet(img)

        cls_loss = criterionCls(output_s, target)
        st_loss  = criterionST(F.log_softmax(output_s/args.T, dim=1),
                               F.softmax(output_t/args.T, dim=1)) * (args.T*args.T) / img.size(0)
        st_loss  = st_loss * args.lambda_st

        prec1, prec5 = accuracy(output_s, target, topk=(1,5))
        cls_losses.update(cls_loss.item(), img.size(0))
        st_losses.update(st_loss.item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))

    f_l = [cls_losses.avg, st_losses.avg, top1.avg, top5.avg]
    print('Cls: {:.4f}, ST: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l))
    log_list.append('Cls: {:.4f}, ST: {:.4f}, Prec@1: {:.2f}, Prec@5: {:.2f}\n'.format(*f_l))
    with open('./log_test_cinic.txt', 'a+') as f:
        f.writelines(log_list)

def adjust_lr(optimizer, epoch):
    scale   = 0.1
    # lr_list =  [args.lr] * 100
    # lr_list += [args.lr*scale] * 50
    # lr_list += [args.lr*scale*scale] * 50

    lr_list = [args.lr] * 200

    lr_list += [args.lr] * 60
    lr_list += [args.lr * scale] * 30
    lr_list += [args.lr * scale * scale] * 10


    lr = lr_list[epoch-1]
    print('epoch: {}  lr: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()