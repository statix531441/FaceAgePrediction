import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import load_history_model
from data import Dataset, create_dataloaders



parser = argparse.ArgumentParser(description='Testing models')

parser.add_argument('--model_name', type=str, default='dnet121-pt', help='Start training from an existing model.')
parser.add_argument('--best', dest='best', action='store_true', help='Load best model weights')



args = parser.parse_args()


model_name = args.model_name
checkpoint_path = os.path.join('checkpoints', model_name)
checkpoint_best = os.path.join(checkpoint_path, 'best.pt')
checkpoint_latest = os.path.join(checkpoint_path, 'latest.pt')
train_args_path = os.path.join(checkpoint_path, 'train_args.txt')

with open(train_args_path, 'r') as f:
    if args.best:
        args = argparse.Namespace(**json.load(f))
        args.checkpoint = checkpoint_best
    else:
        args = argparse.Namespace(**json.load(f))
        args.checkpoint = checkpoint_latest
    
print(args)

start_epoch, history, model = load_history_model(args)

train = pd.read_csv(os.path.join(args.csv_folder, 'train.csv'))
test = pd.read_csv(os.path.join(args.csv_folder, 'test.csv'))

if torch.cuda.is_available(): device = 'cuda:0'
else: device = 'cpu'
print(f"{device} used for testing.")

print(f"Test accuracy from history: {history['test_accuracy'][-1]}")

test_set = Dataset(test, num_classes=args.num_classes)
test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False)
test_acc = 0
with torch.no_grad():
    model.to(device)
    model.eval()
    for batch_idx, (X, y, age_cat) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)

        oh = model(X).softmax(dim=1)
        test_acc += (oh.argmax(dim=1) == y.argmax(dim=1)).float().sum() 

    test_acc /= len(test_loader.dataset)
    test_acc = test_acc.to('cpu').item()

print(f"Test accuracy evaluated: {test_acc}")