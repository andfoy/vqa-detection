# -*- coding: utf-8 -*-

"""
Visual Genome Dataloader.

This file contains the implementation of a PyTorch compilant dataset
to load Visual Genome Images, Region descriptions, bounding boxes and
VQA.
"""

import cv2
import json
import torch
import progressbar
import numpy as np
import os.path as osp
import torch.utils.data as data
import visual_genome.local as vg


class VGLoader(data.Dataset):
    """Visual Genome dataset PyTorch loader."""
    TRAIN_FILE = 'vg_train.pth'
    TEST_FILE = 'vg_test.pth'
    VAL_FILE = 'vg_val.pth'
    OBJ_IDX_FILE = 'vg_classes.pth'
    DATA_FOLDER = 'data'
    NUM_CLASSES = 50
    NUM_OBJS = 2
    SPLIT_PROPORTION = 0.33

    def __init__(self, data_root, transform=None,
                 target_transform=None, train=True, test=False):
        """Dataset main constructor."""
        self.images = []
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

        train_file_path = osp.join(self.DATA_FOLDER, self.TRAIN_FILE)
        val_file_path = osp.join(self.DATA_FOLDER, self.VAL_FILE)
        test_file_path = osp.join(self.DATA_FOLDER, self.TEST_FILE)
        obj_idx_file_path = osp.join(self.DATA_FOLDER, self.OBJ_IDX_FILE)

        if self.train:
            self.images = torch.load(train_file_path)
        elif self.test:
            self.images = torch.load(test_file_path)
        else:
            self.images = torch.load(val_file_path)

        self.obj_idx = torch.load(obj_idx_file_path)

    def check_exists(self):
        """Check if dataset has been processed before."""
        return osp.exists(osp.join(self.DATA_FOLDER, self.TRAIN_FILE))

    def filter_regions(self):
        """Choose top NUM_CLASSES categories that contain NUM_OBJS objects."""
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

        print('Processing region graph objects: Extract top {0} categories '
              'with {1} objects'.format(
                  self.NUM_CLASSES, self.NUM_OBJS))

        obj_count = {}
        bar = progressbar.ProgressBar()
        for img in bar(img_id):
            for region in img_id[img]:
                obj = img_id[img][region]
                if len(obj) == self.NUM_OBJS:
                    if obj not in obj_count:
                        obj_count[obj] = 0
                    obj_count[obj] += 1

        objs = sorted(obj_count, key=lambda k: obj_count[k],
                      reverse=True)[:self.NUM_CLASSES]

        obj_idx = dict(zip(objs, range(len(objs))))

        print('Filtering regions...')

        img_regions = {}
        bar = progressbar.ProgressBar()
        for img in bar(img_id):
            regions = {}
            for region in img_id[img]:
                if img_id[img][region] in obj_idx:
                    regions[region] = img_id[img][region]
            if len(regions) > 0:
                img_regions[img] = regions

        return obj_idx, img_regions

    def process_dataset(self):
        """Load, transform and split dataset."""
        obj_idx, img_regions = self.filter_regions()
        num_images = len(img_regions)

        print('Loading region bounding boxes...')
        region_descriptions = vg.get_all_region_descriptions(self.data_root)
        bar = progressbar.ProgressBar()
        for region_group in bar(region_descriptions):
            for region in region_group:
                if region.image.id in img_regions:
                    if region.id in img_regions[region.image.id]:
                        cat = img_regions[region.image.id][region.id]
                        img_regions[region.image.id][region.id] = (region, cat)

        print('Splitting dataset...')
        num_images_split = int(np.ceil(num_images * self.SPLIT_PROPORTION))

        image_id = np.array(img_regions.keys())
        idx_perm = np.random.permutation(num_images)

        train_id = image_id[idx_perm[:num_images_split]]
        val_id = image_id[idx_perm[num_images_split:num_images_split * 2]]
        test_id = image_id[idx_perm[num_images_split * 2:]]

        train_images = [img_regions[img] for img in train_id]
        val_images = [img_regions[img] for img in val_id]
        test_images = [img_regions[img] for img in test_id]

        train_file_path = osp.join(self.DATA_FOLDER, self.TRAIN_FILE)
        val_file_path = osp.join(self.DATA_FOLDER, self.VAL_FILE)
        test_file_path = osp.join(self.DATA_FOLDER, self.TEST_FILE)
        obj_idx_file_path = osp.join(self.DATA_FOLDER, self.OBJ_IDX_FILE)

        print('Saving data...')
        torch.save(train_images, train_file_path)
        torch.save(val_images, val_file_path)
        torch.save(test_images, test_file_path)
        torch.save(obj_idx, obj_idx_file_path)

    def __len__(self):
        """Number of dataset images."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get image at index"""
        img_regions = self.images[idx]
        img_info = img_regions[0].image

        uri = img_info.url
        folder, filename = uri.split('/')[-2:]
        img_path = osp.join(self.data_root, folder, filename)

        img = cv2.imread(img_path)
        height, width, channels = img.shape

        if self.target_transform is not None:
            img_regions = self.target_transform(img_regions, width, height)

        if self.transform is not None:
            img_regions = np.array(img_regions)
            img, boxes, labels = self.transform(
                img, img_regions[:, :4], img_regions[:, 4])

            img = img[:, :, (2, 1, 0)]
            img_regions = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return (torch.from_numpy(img).permute(2, 0, 1), img_regions,
                height, width)
