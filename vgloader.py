
import os
import cv2
import json
import torch
import numpy as np
import progressbar2
import os.path as osp
import torch.utils.data as data
import visual_genome.local as vg


class VGLoader(data.Dataset):
    """Visual Genome dataset PyTorch loader."""
    TRAIN_FILE = 'vg_train.pth'
    TEST_FILE = 'vg_test.pth'
    VAL_FILE = 'vg_val.pth'
    DATA_FOLDER = 'data'
    NUM_CLASSES = 50
    NUM_OBJS = 2

    def __init__(self, data_root, transform=None,
                 target_transform=None, train=True, test=False):
        """Dataset main constructor."""
        self.data_root = data_root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.test = test
        self.validation = not train and not test

        if not osp.exists(self.data_root):
            raise RuntimeError('Dataset not found ' +
                               'please download it from: ' +
                               'http://visualgenome.org/api/v0/api_home.html')

        if not self.check_exists():
            self.process_dataset()

    def check_exists(self):
        return osp.exists(osp.join(self.DATA_FOLDER, self.TRAIN_FILE))

    def filter_regions(self):
        print('Loading region graph objects...')
        region_graph_file = osp.join(self.data_root, 'region_graphs.json')
        with open(region_graph_file, 'r') as f:
            reg_graph = json.load(f)

        img_id = {x['image_id']: {
            y['region_id']: frozenset([z['entity_name'].lower()
                                       for z in y['synsets']] +
                                      [z['name'].lower()
                                       for z in y['objects']])
            for y in x['regions']}
            for x in reg_graph}

        print('Processing region graph objects: Extract top {0} categories'
              'with {1} objects'.format(
                  self.NUM_CLASSES, self.NUM_OBJS))

        obj_count = {}
        bar = progressbar2.ProgressBar()
        for img in bar(img_id):
            for region in img_id[img]:
                for obj in img_id[img][region]:
                    if len(obj) == self.NUM_OBJS:
                        if obj not in obj_count:
                            obj_count[obj] = 0
                        obj_count[obj] += 1

        objs = sorted(obj_count, key=lambda k: obj_count[k],
                      reverse=True)[:self.NUM_CLASSES]

        obj_idx = dict(zip(objs, range(len(objs))))

        print('Filtering regions...')

        img_regions = {}
        bar = progressbar2.ProgressBar()
        for img in bar(img_id):
            regions = {}
            for region in img_id[img]:
                if img_id[img][region] in obj_idx:
                    regions[region] = img_id[img][region]
            if len(regions) > 0:
                img_regions[img] = regions

        return obj_idx, img_regions

    def process_dataset(self):
        pass
