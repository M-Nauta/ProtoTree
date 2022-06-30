import argparse
import torch.nn as nn
import torch.nn.functional as F
from prototree.prototree import ProtoTree
from util.log import Log
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
from features.densenet_features import densenet121_features, densenet161_features, densenet169_features, densenet201_features
from features.vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,vgg19_features, vgg19_bn_features

base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'densenet161': densenet161_features,
                                 'densenet169': densenet169_features,
                                 'densenet201': densenet201_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features}

"""
    Create network with pretrained features and 1x1 convolutional layer

"""
def get_network(num_in_channels: int, args: argparse.Namespace):
    # Define a conv net for estimating the probabilities at each decision node
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)            
    features_name = str(features).upper()
    if features_name.startswith('VGG') or features_name.startswith('RES'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
    elif features_name.startswith('DENSE'):
        first_add_on_layer_in_channels = \
            [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
    else:
        raise Exception('other base base_architecture NOT implemented')
    
    add_on_layers = nn.Sequential(
                    nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=args.num_features, kernel_size=1, bias=False),
                    nn.Sigmoid()
                    ) 
    return features, add_on_layers

def freeze(tree: ProtoTree, epoch: int, params_to_freeze: list, params_to_train: list, args: argparse.Namespace, log: Log):
    if args.freeze_epochs>0:
        if epoch == 1:
            log.log_message("\nNetwork frozen")
            for parameter in params_to_freeze:
                parameter.requires_grad = False
        elif epoch == args.freeze_epochs + 1:
            log.log_message("\nNetwork unfrozen")
            for parameter in params_to_freeze:
                parameter.requires_grad = True

