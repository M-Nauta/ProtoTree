from shutil import copy
from copy import deepcopy
import torch
import os
import numpy as np
from prototree.prototree import ProtoTree
from util.log import Log
from util.args import get_args, save_args, get_optimizer
from util.data import get_dataloaders
from util.analyse import analyse_ensemble

import gc

from main_tree import run_tree

def run_ensemble():
    all_args = get_args()
    # Create a logger
    log = Log(all_args.log_dir)
    print("Log dir: ", all_args.log_dir, flush=True)
    # Log the run arguments
    save_args(all_args, log.metadata_dir)
    
    if not all_args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.isdir(os.path.join(all_args.log_dir, "files")):
        os.mkdir(os.path.join(all_args.log_dir, "files")) 

    # Obtain the data loaders
    trainloader, projectloader, test_loader, classes, num_channels = get_dataloaders(all_args)

    log_dir_orig = all_args.log_dir

    trained_orig_trees = []
    trained_pruned_trees = []
    trained_pruned_projected_trees = []
    orig_test_accuracies = []
    pruned_test_accuracies = []
    pruned_projected_test_accuracies = []
    project_infos = []
    infos_sample_max = []
    infos_greedy = []
    infos_fidelity = []
    # Train trees in ensemble one by one and save corresponding trees and accuracies
    for pt in range(1,all_args.nr_trees_ensemble+1):
        torch.cuda.empty_cache()
        
        print("\nTraining tree ",pt, "/", all_args.nr_trees_ensemble, flush=True)
        log.log_message('Training tree %s...'%str(pt))

        args = deepcopy(all_args)
        args.log_dir = os.path.join(log_dir_orig,'tree_'+str(pt))

        trained_tree, pruned_tree, pruned_projected_tree, original_test_acc, pruned_test_acc, pruned_projected_test_acc, project_info, eval_info_samplemax, eval_info_greedy, info_fidelity = run_tree(args)

        trained_orig_trees.append(trained_tree)
        trained_pruned_trees.append(pruned_tree)
        trained_pruned_projected_trees.append(pruned_projected_tree)
    
        orig_test_accuracies.append(original_test_acc)
        pruned_test_accuracies.append(pruned_test_acc)
        pruned_projected_test_accuracies.append(pruned_projected_test_acc)
        
        project_infos.append(project_info)
        infos_sample_max.append(eval_info_samplemax)
        infos_greedy.append(eval_info_greedy)
        infos_fidelity.append(info_fidelity)
    
        if pt > 1:
            #analyse ensemble with > 1 trees:
            analyse_ensemble(log, all_args, test_loader, device, trained_orig_trees, trained_pruned_trees, trained_pruned_projected_trees, orig_test_accuracies, pruned_test_accuracies, pruned_projected_test_accuracies, project_infos, infos_sample_max, infos_greedy, infos_fidelity)
            
if __name__ == '__main__':
    run_ensemble()
