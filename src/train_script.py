from __future__ import print_function

import os
os.chdir(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import draw_bounding_boxes
# from torchvision.transforms import ConvertImageDtype
import torchvision.models.detection as detection

import copy

from dataloader import CocoDataset

import pandas as pd
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import json

from rcnn_utils.engine import train_one_epoch, evaluate
from rcnn_utils import utils as utils

import argparse
import models

def sample_transform(sample):
    # convert channel last to channel first format
    img = torch.tensor(sample['img'].transpose(2, 0, 1))
    boxes = torch.tensor(sample['annot'][:, :4])
    labels = torch.tensor(sample['annot'][:, 4].astype(int))
    img_id = torch.tensor(sample['img_id'])

    # convert [x, y, w, h] to [x_min, y_min, x_max, y_max]
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]


    annot = {'boxes': boxes, 'labels': labels, 'image_id': img_id}
    return (img, annot)

def train_and_eval(model, dataset, dataset_test, *, epochs=2, checkpoint=None, save_path=None, device='cpu', save_every=1):
    if save_path != None:
        os.makedirs(save_path, mode=777, exist_ok=True)
        assert os.path.isdir(save_path), '{} is not a directory'.format(save_path)
    start_epoch = 0

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)


    # construct an optimizer
    # and a learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.001)


    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                  step_size=3,
                                                  gamma=0.45)


    if checkpoint != None:
        print('load stuff')
        ckpt = torch.load(checkpoint, map_location=device)

        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        lr_scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch']

        del ckpt
    else:
        print('not load stuff')
    

    # let's train it for 10 epochs
    num_epochs = epochs
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, start_epoch + epoch, print_freq=500)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, coco_dataset.coco, data_loader_test, device=device)
        
        # saving checkpoint
        if (start_epoch + epoch + 1) % save_every == 0 and save_path != None:
            print("Saving checkpoint...")
            save_path_final = os.path.join(save_path, 'epoch_{}.pth'.format(start_epoch + epoch + 1))
            torch.save({'epoch': start_epoch + epoch + 1,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict()}, save_path_final)

    # final save after training
    if (start_epoch + num_epochs) % save_every != 0 and save_path != None:
        save_path_final = os.path.join(save_path, 'epoch_{}.pth'.format(num_epochs))
        torch.save({'epoch': start_epoch + num_epochs,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict()}, save_path_final)

###################################
# Script start
parser = argparse.ArgumentParser()
parser.add_argument('-l', '--load', type=str, default=None, help='specify a checkpoint file to resume training from')
parser.add_argument('-s', '--save', type=str, default=None, help='specify a directory to save a checkpoint at the end of training')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
parser.add_argument('--save_every', type=int, default=5, metavar='SE', help='save every # of epochs')
parser.add_argument('-i', '--imp_mode', type=str, default='111', metavar='MASK', help='a three bit mask to indicate which implicit implementation is active')
args = parser.parse_args()
print(args)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

assert len(args.imp_mode) == 3, 'Mode mask needs to be a 3 bit mask'
imp_mode = [c == '1' for c in args.imp_mode]
model = models.model_imp(*imp_mode).to(device)

coco_dataset = CocoDataset('../miniCOCO', 'minitrain2017', transform=sample_transform)

# split the dataset in train and test set
indices = torch.load('permutated_indices.pt').tolist()
dataset = torch.utils.data.Subset(coco_dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(coco_dataset, indices[-50:])

train_and_eval(model, dataset, dataset_test, epochs=args.epochs, device=device, checkpoint=args.load, save_path=args.save, save_every=args.save_every)


