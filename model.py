import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


def load_history_model(args):
    
    model = AgePredictor(feature_extractor=args.feature_extractor, pretrained=args.pretrained)
    print(f"Model Loaded with feature extractor={args.feature_extractor} and pretrained={args.pretrained}")
    history = {'loss':[],
            'train_accuracy':[],
            'test_accuracy':[]}
    start_epoch = 1

    if args.checkpoint:
        state = torch.load(args.checkpoint)
        start_epoch, history, model_state_dict, optimizer_state_dict = state['epoch'], state['history'], state['model_state_dict'], state['optimizer_state_dict']
        model.load_state_dict(model_state_dict)
        print(f"Model weights loaded from {args.checkpoint}")

    return start_epoch, history, model




class AgePredictor(nn.Module):
    def __init__(self, feature_extractor = 'dnet121', pretrained=True, num_classes=10):
        super(AgePredictor, self).__init__()
        
        if feature_extractor == 'dnet121':
            self.features = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1).features
        elif feature_extractor == 'dnet169':
            self.features = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1).features
        
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