import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


def load_history_model(args):
    
    model = AgePredictor(feature_extractor=args.feature_extractor, pretrained=args.pretrained, num_classes=args.num_classes)
    print(f"Model Loaded with feature extractor={args.feature_extractor}, num_classes={args.num_classes}, pretrained={args.pretrained} and freeze_extractor={args.freeze_extractor}")
    history = {'loss':[],
            'train_accuracy':[],
            'test_accuracy':[]}
    start_epoch = 0

    if args.checkpoint:
        if torch.cuda.is_available(): device = 'cuda:0'
        else: device = 'cpu'
        state = torch.load(args.checkpoint, map_location=torch.device(device))
        start_epoch, history, model_state_dict, optimizer_state_dict = state['epoch'], state['history'], state['model_state_dict'], state['optimizer_state_dict']
        model.load_state_dict(model_state_dict)
        print(f"Model weights loaded from {args.checkpoint} into {device}")

    return start_epoch, history, model




class AgePredictor(nn.Module):
    def __init__(self, feature_extractor = 'dnet121', pretrained=True, num_classes=10):
        super(AgePredictor, self).__init__()
        
        if feature_extractor == 'dnet121':
            if pretrained: self.features = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
            else: self.features = models.densenet121().features
        elif feature_extractor == 'dnet169':
            if pretrained: self.features = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1).features
            else: self.features = models.densenet169().features
        
        feature_vector_size = self.features.norm5.num_features
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_vector_size, 512),
            nn.Sigmoid(),
            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.Linear(256, num_classes)
        )
        
        
    def forward(self, x):
        
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out