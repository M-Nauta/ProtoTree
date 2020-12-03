import argparse
import torch
from prototree.prototree import ProtoTree
import os
import pickle

def load_state(directory_path: str, device):
    with open(directory_path + '/tree.pkl', 'rb') as f:
        tree = pickle.load(f)
        state = torch.load(directory_path + '/model_state.pth', map_location=device)
        tree.load_state_dict(state)
    return tree

def init_tree(tree: ProtoTree, optimizer, scheduler, device, args: argparse.Namespace):
    epoch = 1
    mean = 0.5
    std = 0.1
    # load trained prototree if flag is set

    # NOTE: TRAINING FURTHER FROM A CHECKPOINT DOESN'T SEEM TO WORK CORRECTLY. EVALUATING A TRAINED PROTOTREE FROM A CHECKPOINT DOES WORK. 
    if args.state_dict_dir_tree != '':
        if not args.disable_cuda and torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
        else:
            device = torch.device('cpu')

        
        if args.disable_cuda or not torch.cuda.is_available():
        # tree = load_state(args.state_dict_dir_tree, device)
            tree = torch.load(args.state_dict_dir_tree+'/model.pth', map_location=device)
        else:
            tree = torch.load(args.state_dict_dir_tree+'/model.pth')
        tree.to(device=device)
        try:
            epoch = int(args.state_dict_dir_tree.split('epoch_')[-1]) + 1
        except:
            epoch=args.epochs+1
        print("Train further from epoch: ", epoch, flush=True)
        optimizer.load_state_dict(torch.load(args.state_dict_dir_tree+'/optimizer_state.pth', map_location=device))

        if epoch>args.freeze_epochs:
            for parameter in tree._net.parameters():
                parameter.requires_grad = True
        if not args.disable_derivative_free_leaf_optim:
            for leaf in tree.leaves:
                leaf._dist_params.requires_grad = False
        
        if os.path.isfile(args.state_dict_dir_tree+'/scheduler_state.pth'):
            # scheduler.load_state_dict(torch.load(args.state_dict_dir_tree+'/scheduler_state.pth'))
            # print(scheduler.state_dict(),flush=True)
            scheduler.last_epoch = epoch - 1
            scheduler._step_count = epoch
        

    elif args.state_dict_dir_net != '': # load pretrained conv network
        # initialize prototypes
        torch.nn.init.normal_(tree.prototype_layer.prototype_vectors, mean=mean, std=std)
        #strict is False so when loading pretrained model, ignore the linear classification layer
        tree._net.load_state_dict(torch.load(args.state_dict_dir_net+'/model_state.pth'), strict=False)
        tree._add_on.load_state_dict(torch.load(args.state_dict_dir_net+'/model_state.pth'), strict=False) 
    else:
        with torch.no_grad():
            # initialize prototypes
            torch.nn.init.normal_(tree.prototype_layer.prototype_vectors, mean=mean, std=std)
            tree._add_on.apply(init_weights_xavier)
    return tree, epoch

def init_weights_xavier(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

def init_weights_kaiming(m):
    if type(m) == torch.nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')