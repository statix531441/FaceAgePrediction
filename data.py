import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class Dataset(Dataset):
    def __init__(self, data=None, datafrac=1, imageFolder='UTKFace'):
        
        self.imageFolder = imageFolder
        self.data = data.sample(frac=datafrac, random_state=1).reset_index(drop=True)
        self.preprocess = transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.encoded = F.one_hot(torch.as_tensor(self.data['class'].to_numpy()), 10).to(torch.float)
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.imageFolder, self.data.loc[idx, 'image_path']))
        X = self.preprocess(image)
        y = self.encoded[idx]
        age_category = self.data.loc[idx, 'age_category']
        return X, y, age_category

def create_csv():
    print(f"Creating test-train split and storing in csv")
    age_range = 10
    bins = [x for x in range(0, 100+age_range, age_range)]

    imageFolder = 'UTKFace'
    df = pd.DataFrame({'image_path': os.listdir(imageFolder)})

    df['age'] = df.image_path.apply(lambda x: int(x.split('_')[0]))
    df['age_category'] = pd.cut(df['age'], bins=bins)
    df['class'] = df.age_category.apply(lambda x: int(bins.index(x.left)))
    df['gender'] = df.image_path.apply(lambda x: int(x.split('_')[1]) if len(x.split('_')[1])==1 else np.nan)
    df['ethnicity'] = df.image_path.apply(lambda x: int(x.split('_')[2]) if len(x.split('_')[2])==1 else np.nan)
    df = df.dropna()

    df = df.drop(df[df['class'] == 0][:2000].index, axis=0)
    df = df.drop(df[df['class'] == 2][:5700].index, axis=0)
    df = df.drop(df[df['class'] == 3][:2000].index, axis=0)
    # df = df.drop(df[df['class'] == 10].index, axis=0)

    train = df.sample(frac=0.8, random_state=1)
    test = df.drop(train.index)

    # 80% train-test split
    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)


def create_dataloaders(args):

    train_set = Dataset(pd.read_csv('train.csv'), args.datafrac, imageFolder='UTKFace')
    test_set = Dataset(pd.read_csv('test.csv'), args.datafrac, imageFolder='UTKFace')

    print(f"Train size: {len(train_set)}, Test size: {len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=args.bs, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.bs, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    create_csv()