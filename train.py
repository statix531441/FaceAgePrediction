import os
import argparse
import json
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn

from model import load_history_model
from data import create_dataloaders

parser = argparse.ArgumentParser(description='Age Prediction via Transfer Learning')

# Training Options
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--bs', type=int, default=50, help='Batch size')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
parser.add_argument('--datafrac', type=float, default=1, help='Fraction of dataset used')
parser.add_argument('--freeze_extractor', dest='freeze_extractor', action='store_true', help='Tune weights of feature extractor')

# Model Architecture Options
parser.add_argument('--feature_extractor', type=str, default='dnet121', choices=['dnet121', 'dnet169'], help="Feature Extractor used")
parser.add_argument('--pretrained', dest='pretrained', action='store_true', help='Load pretrained model weights')

# Checkpoint Options
parser.add_argument('--checkpoint', type=str, default=None, help='Start training from an existing model.')
parser.add_argument('--name', type=str, default='dnet121-Tpt-Fx', help='A name to attach to the training session')

args = parser.parse_args()

if args.checkpoint:
    train_args_path = os.path.join(os.path.dirname(args.checkpoint),'train_args.txt')
    with open(train_args_path, 'r') as f:
        chk_args = json.load(f)
    args.feature_extractor = chk_args['feature_extractor']


if torch.cuda.is_available(): device = 'cuda:0'
else: device = 'cpu'
print(f"{device} used for training.")


train_loader, test_loader = create_dataloaders(args)

start_epoch, history, model = load_history_model(args)

criterion = nn.CrossEntropyLoss()

if args.freeze_extractor:
    for p in list(model.features.parameters()):
        p.requires_grad = False
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)



model.to(device)
criterion.to(device)

checkpoints_folder = os.path.join('checkpoints', args.name)
if not os.path.exists(checkpoints_folder):
    os.makedirs(checkpoints_folder)

best_path = os.path.join(checkpoints_folder, 'best.pt')
latest_path = os.path.join(checkpoints_folder, 'latest.pt')

with open(os.path.join(checkpoints_folder, 'train_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)





# Training

for epoch in range(start_epoch, start_epoch+args.epochs):
    train_acc = 0
    test_acc = 0

    model.train()
    for batch_idx, (X, y, age_cat) in tqdm(enumerate(train_loader), total=len(train_loader), ascii=True, desc=f"Epoch {epoch}"):
        # print(f"batch {batch_idx}", end=': ')
        X, y = X.to(device), y.to(device)
        
        out = model(X)
        oh = out.softmax(dim=1)
        
        loss = criterion(oh, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.to('cpu').item() # transfer to cpu and extract value before appending
        # print(f"loss: {loss}")
        history['loss'].append(loss) 
        
        # Training accuracy
        train_acc += (oh.argmax(dim=1) == y.argmax(dim=1)).float().sum()
        
    train_acc /= len(train_loader.dataset)
    train_acc = train_acc.to('cpu').item() # transfer to cpu and extract value before appending
    print(f"train_acc: {train_acc}", end=", ")
    history['train_accuracy'].append(train_acc)  
    
    # Testing accuracy
    with torch.no_grad():
        model.eval()
        for batch_idx, (X, y, age_cat) in enumerate(test_loader):
            X, y = X.to(device), y.to(device)

            oh = model(X).softmax(dim=1)
            test_acc += (oh.argmax(dim=1) == y.argmax(dim=1)).float().sum() 

        test_acc /= len(test_loader.dataset)
        test_acc = test_acc.to('cpu').item()
        print(f"test_acc: {test_acc}", end=" ")
        history['test_accuracy'].append(test_acc) # transfer to cpu and extract value before appending
        
    
    # Save latest model
    torch.save({'epoch': epoch,
               'history': history,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()}, latest_path)
    
    # Save best model (based on test accuracy)
    if len(history['test_accuracy'])==1 or test_acc > max(history['test_accuracy']):
           torch.save({'epoch': epoch,
               'history': history,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict()}, best_path)


visualization_folder = os.path.join('visualization', args.name)
if not os.path.exists(visualization_folder):
    os.makedirs(visualization_folder)

for k, v in history.items():
    plt.figure()
    plt.title(k)
    plt.plot(v)
    plt.savefig(os.path.join(f"{visualization_folder}", k))