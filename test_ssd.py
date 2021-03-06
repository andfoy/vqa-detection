# -*- coding: utf-8 -*-

"""
SSD detection test.
Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

# Standard lib imports
import os
import time
import argparse
import os.path as osp

# PyTorch imports
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

# Local module imports
from ssd.ssd import build_ssd
from vgloader import VGLoader, AnnotationTransform
from ssd.utils.augmentations import Normalize, Compose, Resize

# Other module imports
import pickle
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--model', default='weights/vqa_baseline_imagenet.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--no-cuda', action='store_true',
                    help='Do not use cuda to train model')
parser.add_argument('--data', type=str, default='../visual_genome',
                    help='path to Visual Genome dataset')
parser.add_argument('--num-classes', type=int, default=50,
                    help='number of classification categories')
parser.add_argument('--batch-size', default=16, type=int,
                    help='Batch size for testing')

args = parser.parse_args()

if not osp.exists(args.save):
    os.mkdir(args.save)

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = osp.join(name, phase)
    if not osp.exists(filedir):
        os.makedirs(filedir)
    return filedir


def get_voc_results_file_template(image_set, cls):
    # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
    filename = 'det_' + image_set + '_%s.txt' % (cls)
    filedir = osp.join(args.save, 'results')
    if not osp.exists(filedir):
        os.makedirs(filedir)
    path = osp.join(filedir, filename)
    return path


def write_voc_results_file(all_boxes, dataset):
    labelmap = dataset.obj_idx
    for classname in labelmap:
        cls_ind = labelmap[classname]
        print('Writing {0} results file'.format(str(classname)))
        filename = get_voc_results_file_template('test', cls_ind)
        with open(filename, 'wt') as f:
            dets_cls = all_boxes[cls_ind]
            for im_ind in dets_cls:
                dets = dets_cls[im_ind]
                if dets == []:
                    continue
                # the VOCdevkit expects 1-based indices
                for k in range(dets.shape[0]):
                    f.write('{:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                            format(im_ind, dets[k, -1],
                                   dets[k, 0] + 1, dets[k, 1] + 1,
                                   dets[k, 2] + 1, dets[k, 3] + 1))


def do_python_eval(labelmap, output_dir='output', use_07=True):
    aps = []
    # The PASCAL VOC metric changed in 2010
    use_07_metric = use_07
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not osp.isdir(output_dir):
        os.mkdir(output_dir)
    for classname in labelmap:
        cls_ind = labelmap[classname]
        # filename = get_voc_results_file_template('test', cls_ind)
        rec, prec, ap = voc_eval(
            cls_ind, ovthresh=0.5, use_07_metric=use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(classname, ap))
        with open(osp.join(output_dir, str(cls_ind) + '_pr.pkl'), 'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('Mean AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~')
    print('')
    print('--------------------------------------------------------------')
    print('Results computed with the **unofficial** Python eval code.')
    print('Results should be very close to the official MATLAB eval code.')
    print('--------------------------------------------------------------')


def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(classname,
             ovthresh=0.5,
             use_07_metric=True):
    """rec, prec, ap = voc_eval(detpath,
                           annopath,
                           imagesetfile,
                           classname,
                           [ovthresh],
                           [use_07_metric])
Top level function that does the PASCAL VOC evaluation.
detpath: Path to detections
   detpath.format(classname) should produce the detection results file.
annopath: Path to annotations
   annopath.format(imagename) should be the xml annotations file.
imagesetfile: Text file containing the list of images, one image per line.
classname: Category name (duh)
cachedir: Directory for caching the annotations
[ovthresh]: Overlap threshold (default = 0.5)
[use_07_metric]: Whether to use VOC07's 11 point AP computation
   (default False)
"""
# assumes detections are in detpath.format(classname)
# assumes annotations are in annopath.format(imagename)
# assumes imagesetfile is a text file with each line an image name
# cachedir caches the annotations in a pickle file
# first load gt
    output_dir = get_output_dir('ssd300_vg_50', 'test')
    gt_file = osp.join(output_dir, 'ground_truth.pth')

    # if not osp.isdir(cachedir):
    #     os.mkdir(cachedir)
    # cachefile = osp.join(cachedir, 'annots.pkl')
    # # read list of images
    # with open(imagesetfile, 'r') as f:
    #     lines = f.readlines()
    # imagenames = [x.strip() for x in lines]
    # if not osp.isfile(cachefile):
    #     # load annots
    #     recs = {}
    #     for i, imagename in enumerate(imagenames):
    #         recs[imagename] = parse_rec(annopath % (imagename))
    #         if i % 100 == 0:
    #             print('Reading annotation for {:d}/{:d}'.format(
    #                i + 1, len(imagenames)))
    #     # save
    #     print('Saving cached annotations to {:s}'.format(cachefile))
    #     with open(cachefile, 'wb') as f:
    #         pickle.dump(recs, f)
    # else:
    #     # load
    #     with open(cachefile, 'rb') as f:
    #         recs = pickle.load(f)
    recs = torch.load(gt_file)
    recs = recs[classname]

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in recs:
        R = recs[imagename]
        bbox = np.array(R)
        difficult = np.zeros(len(R)).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets
    detfile = get_voc_results_file_template('test', classname)
    print(detfile)
    with open(detfile, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:

        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [int(x[0]) for x in splitlines]
        confidence = np.array([float(x[1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        # go down dets and mark TPs and FPs
        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d in range(nd):
            if image_ids[d] not in class_recs:
                continue
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > ovthresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.

    return rec, prec, ap


def test_net(dataset):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(dataset)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = {x: {} for x in range(len(dataset.obj_idx))}
    cls_gt = {i: {} for i in range(len(dataset.obj_idx))}

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    output_dir = get_output_dir('ssd300_vg_50', 'test')
    det_file = osp.join(output_dir, 'detections.pth')
    gt_file = osp.join(output_dir, 'ground_truth.pth')

    if not osp.exists(det_file):
        for i in range(len(dataset)):
            img, boxes, h, w, img_id = dataset[i]
            x = Variable(img.unsqueeze(0))
            if args.cuda:
                x = x.cuda()
            _t['im_detect'].tic()
            detections = net(x).data
            detect_time = _t['im_detect'].toc(average=False)

            # boxes = [[x[y] * d for y, d in zip(range(0, 4), (w, h, w, h))]
            #          for x in boxes]
            for box in boxes:
                cat = box[4]
                box[0] *= w
                box[2] *= w
                box[1] *= h
                box[3] *= h
                box = box[:-1]
                if img_id not in cls_gt[cat]:
                    cls_gt[cat][img_id] = []
                cls_gt[cat][img_id].append(box)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.dim() == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((
                    boxes.cpu().numpy(), scores[:, np.newaxis])).astype(
                        np.float32, copy=False)
                all_boxes[j - 1][img_id] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s'.format(
                i + 1, num_images, detect_time))

        torch.save(all_boxes, det_file)
        torch.save(cls_gt, gt_file)
        write_voc_results_file(all_boxes, dataset)

    print('Evaluating detections')
    do_python_eval(dataset.obj_idx, output_dir=output_dir)


if __name__ == '__main__':
    # load net
    num_classes = args.num_classes + 1  # +1 background
    net = build_ssd('test', 300, num_classes)  # initialize SSD
    net.load_state_dict(torch.load(args.model))
    net.eval()
    print('Finished loading model!')
    # load data
    dataset = VGLoader(data_root=args.data,
                       transform=Compose([Resize(size=300),
                                          Normalize(
                                              mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
                                          ]),
                       target_transform=AnnotationTransform(),
                       train=False)
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(dataset)
