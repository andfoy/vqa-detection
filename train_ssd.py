# -*- coding: utf-8 -*-

"""
Train vqa-detection model (SSD component) on Visual Genome dataset.
"""

# Standard lib imports
import os
import time
import argparse
import os.path as osp
from urllib.parse import urlparse
from urllib.request import urlretrieve

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

# Local module imports
from ssd.ssd import build_ssd
from ssd.layers import MultiBoxLoss
from ssd.utils.augmentations import SSDAugmentation
from vgloader import VGLoader, detection_collate, AnnotationTransform

# Other imports
from utils import VisdomWrapper, reporthook


PASCAL_WEIGHTS_URL = ('https://s3.amazonaws.com/amdegroot-models'
                      '/ssd300_mAP_77.43_v2.pth')

parser = argparse.ArgumentParser(description='Single Shot MultiBox '
                                             'Detector for object '
                                             'detection training')
parser.add_argument('--data', type=str, default='../visual_genome',
                    help='path to Visual Genome dataset')
parser.add_argument('--jaccard-threshold', default=0.5, type=float,
                    help='Min Jaccard index for matching')
parser.add_argument('--batch-size', default=16, type=int,
                    help='Batch size for training')
parser.add_argument('--num-workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save-folder', default='weights/',
                    help='Location to save checkpoint models')
parser.add_argument('--num-classes', type=int, default=50,
                    help='number of classification categories')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--backup-iters', type=int, default=1000,
                    help='iteration interval to perform state backups')
parser.add_argument('--save', type=str, default='ssd.pth',
                    help='location to SSD state dict file')
parser.add_argument('--parallel', action='store_true',
                    help='train SSD over multiple GPUs')
parser.add_argument('--visdom', type=str, default=None,
                    help='Visdom URL endpoint')

args = parser.parse_args()

PASCAL_WEIGHTS = osp.join(
    args.save_folder, osp.basename(urlparse(PASCAL_WEIGHTS_URL).path))

args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': args.num_workers,
          'pin_memory': True} if args.cuda else {}

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


global_lr = args.lr
decay_step = 0

num_classes = args.num_classes + 1
ssd_dim = 300
batch_size = args.batch_size

print('Loading train set...')

train_loader = VGLoader(data_root=args.data, transform=SSDAugmentation(),
                        target_transform=AnnotationTransform(),
                        train=True)

print('Loading validation set...')

val_loader = VGLoader(data_root=args.data,
                      target_transform=AnnotationTransform(),
                      train=False, test=False)

if not osp.exists(args.save_folder):
    os.makedirs(args.save_folder)

net = build_ssd('train', ssd_dim, num_classes)
snapshot_file = osp.join(args.save_folder, args.save)

if not osp.exists(snapshot_file):
    if not osp.exists(PASCAL_WEIGHTS):
        print('Downloading pretrained SSD weights...')
        urlretrieve(PASCAL_WEIGHTS_URL, PASCAL_WEIGHTS, reporthook)
    snapshot_file = PASCAL_WEIGHTS

net.load_weights(snapshot_file)

if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    net.cuda()

if args.parallel:
    net = nn.DataParallel(net)

if args.visdom is not None:
    visdom_url = urlparse(args.visdom)

    port = 80
    if visdom_url.port is not None:
        port = visdom_url.port

    print('Initializing Visdom frontend at: {0}:{1}'.format(
          args.visdom, port))
    vis = VisdomWrapper(server=visdom_url.geturl(), port=port)

    vis.init_line_plot('iteration_plt', xlabel='Iteration', ylabel='Loss',
                       title='Current SSD Training Loss',
                       legend=['Loc Loss', 'Conf Loss', 'Loss'])

    vis.init_line_plot('epoch_plt', xlabel='Epoch', ylabel='Loss',
                       title='Epoch SSD Training Loss',
                       legend=['Loc Loss', 'Conf Loss', 'Loss'])

    vis.init_line_plot('val_plt', xlabel='Epoch', ylabel='Loss',
                       title='Validation SSD Training Loss',
                       legend=['Loc Loss', 'Conf Loss', 'Loss'])

trainset = DataLoader(train_loader, shuffle=True, collate_fn=detection_collate,
                      batch_size=args.batch_size)

valset = DataLoader(val_loader, shuffle=False, collate_fn=detection_collate,
                    batch_size=args.batch_size)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False)


def train(epoch, global_lr):
    net.train()
    loc_loss = epoch_loc_loss = 0
    conf_loss = epoch_conf_loss = 0
    total_loss = epoch_total_loss = 0
    start_time = time.time()

    for batch_idx, (imgs, targets) in enumerate(trainset):
        if args.cuda:
            imgs = Variable(imgs.cuda())
            targets = [Variable(x.cuda(), volatile=True) for x in targets]

        optimizer.zero_grad()

        out = net(imgs)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()
        optimizer.step()

        total_loss += loss.data[0]
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]

        epoch_total_loss += total_loss
        epoch_loc_loss += loc_loss
        epoch_conf_loss += conf_loss

        if args.visdom is not None:
            vis.plot_line('iteration_plt',
                          X=torch.ones((1, 3)).cpu() * (batch_idx + epoch),
                          Y=torch.Tensor([loss_l.data[0], loss_c.data[0],
                                          loss_l.data[0] + loss_c.data[0]
                                          ]).unsqueeze(0).cpu(),
                          update='append')

        if batch_idx % args.backup_iters == 0:
            backup_file = osp.join(
                args.save_folder, '{0}-{1}-{2}'.format(
                    epoch, batch_idx, args.save))
            torch.save(net.state_dict(), backup_file)

        if batch_idx % args.log_interval == 0:
            elapsed_time = time.time() - start_time
            cur_total_loss = total_loss / args.log_interval
            cur_loc_loss = loc_loss / args.log_interval
            cur_conf_loss = conf_loss / args.log_interval

            # '| loc loss {:.6f} | conf loss: {:.6f}'
            print('| epoch {:5d} | {:5d}/{:5d} batches '
                  '| ms/batch {:.6f} | total loss {:.6f} '
                  '| loc loss {:.6f} | conf loss: {:.6f} '
                  '| lr {:.6f}'.format(
                      epoch, batch_idx, len(trainset), elapsed_time * 1000,
                      cur_total_loss, cur_loc_loss, cur_conf_loss, global_lr))

            total_loss = 0
            loc_loss = 0
            conf_loss = 0
            start_time = time.time()

    epoch_total_loss /= len(trainset)
    epoch_loc_loss /= len(trainset)
    epoch_conf_loss /= len(trainset)

    if args.visdom is not None:
        vis.plot_line('epoch_plt',
                      X=torch.ones((1, 3)).cpu() * epoch,
                      Y=torch.Tensor([epoch_loc_loss, epoch_conf_loss,
                                      epoch_total_loss
                                      ]).unsqueeze(0).cpu(),
                      update='append')


def validate(epoch):
    net.eval()

    loc_loss = 0
    conf_loss = 0
    total_loss = 0
    start = time.time()

    for batch_idx, (imgs, targets) in enumerate(valset):
        if args.cuda:
            imgs = Variable(imgs.cuda())
            targets = [Variable(x.cuda()) for x in targets]
        out = net(imgs)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c

        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        total_loss += loss.data[0]

    loc_loss /= len(valset)
    conf_loss /= len(valset)
    total_loss /= len(valset)
    total_time = time.time() - start

    if args.visdom is not None:
        vis.plot_line('epoch_plt',
                      X=torch.ones((1, 3)).cpu() * epoch,
                      Y=torch.Tensor([loc_loss, conf_loss,
                                      total_loss
                                      ]).unsqueeze(0).cpu(),
                      update='append')

    return total_loss, total_time


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed
       by 10 at every specified step
       Adapted from PyTorch Imagenet example:
       https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    best_val_loss = None
    try:
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            train(epoch, global_lr)
            val_loss, val_time = validate(epoch)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| valid loss {:.6f} | val time: {:5.2f}s'.format(
                      epoch, time.time() - epoch_start_time, val_loss,
                      val_time))
            print('-' * 89)
            if best_val_loss is None or val_loss < best_val_loss:
                file_name = osp.join(args.save_folder, args.save)
                if args.parallel:
                    state_dict = net.module.state_dict()
                else:
                    state_dict = net.state_dict()
                torch.save(state_dict, file_name)
            else:
                global_lr = adjust_learning_rate(optimizer, args.gamma, epoch)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
