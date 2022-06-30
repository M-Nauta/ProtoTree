import torch
import torch.nn.functional as F
import numpy as np
import argparse
from torch.utils.data import DataLoader
from prototree.prototree import ProtoTree
from util.log import Log
from prototree.test import eval_ensemble

def get_avg_path_length(tree: ProtoTree, info: dict, log: Log):
    # If possible, get the depth of the leaf corresponding to the decision
    if 'out_leaf_ix' not in info.keys():
        log.log_message("Soft tree with distributed routing. Path length is always %s across all nodes"%(tree.depth))
    else: #greedy or sample_max routing
        depths = tree.node_depths        
        # Get a dict mapping all node indices to the node objects
        node_ixs = tree.nodes_by_index
        ixs = info['out_leaf_ix']  # Get all indices of the leaves corresponding to the decisions
        pred_depths = [depths[node_ixs[ix]] for ix in ixs]  # Add them to the collection
        avg_depth = sum(pred_depths) / float(len(pred_depths))
        log.log_message("Tree with deterministic routing. Average path length is %s with std %s"%((avg_depth), np.std(pred_depths)))
        log.log_message("Tree with deterministic routing. Longest path has length %s, shortest path has length %s"%((np.max(pred_depths)), np.min(pred_depths)))
    return pred_depths

def log_learning_rates(optimizer, args: argparse.Namespace, log: Log):
    log.log_message("Learning rate net: "+str(optimizer.param_groups[0]['lr']))
    if 'resnet50_inat' in args.net:
        log.log_message("Learning rate block: "+str(optimizer.param_groups[1]['lr']))
        log.log_message("Learning rate net 1x1 conv: "+str(optimizer.param_groups[2]['lr']))
    else:
        log.log_message("Learning rate net 1x1 conv: "+str(optimizer.param_groups[1]['lr']))
    if args.disable_derivative_free_leaf_optim:
        log.log_message("Learning rate prototypes: "+str(optimizer.param_groups[-2]['lr']))
        log.log_message("Learning rate leaves: "+str(optimizer.param_groups[-1]['lr']))
    else:
        log.log_message("Learning rate prototypes: "+str(optimizer.param_groups[-1]['lr']))

def average_distance_nearest_image(project_info: dict, tree: ProtoTree, log: Log, disable_log = False):
    distances = []
    for node, j in tree._out_map.items():
        if node in tree.branches:
            distances.append(project_info[j]['distance'])
            if not disable_log:
                log.log_message("Node %s has nearest distance %s"%(node.index, project_info[j]['distance']))
    if not disable_log:
        log.log_message("Euclidean distances from latent prototypes in tree to nearest image patch: %s"%(str(distances)))
        log.log_message("Average Euclidean distance and standard deviation from latent prototype to nearest image patch: %s, %s"%(str(np.mean(distances)), str(np.std(distances))))
    return distances

def analyse_leaf_distributions(tree: ProtoTree, log: Log):
    # print for experimental purposes
    max_values = []
    for leaf in tree.leaves:
        if leaf._log_probabilities:
            max_values.append(torch.max(torch.exp(leaf.distribution())).item())
        else:
            max_values.append(torch.max(leaf.distribution()).item())
    max_values.sort()
    log.log_message("Max values in softmax leaf distributions: \n"+str(max_values))

def analyse_output_shape(tree: ProtoTree, trainloader: DataLoader, log: Log, device):
    with torch.no_grad():
        # Get a batch of training data
        xs, ys = next(iter(trainloader))
        xs, ys = xs.to(device), ys.to(device)
        log.log_message("Image input shape: "+str(xs[0,:,:,:].shape))
        log.log_message("Features output shape (without 1x1 conv layer): "+str(tree._net(xs).shape))
        log.log_message("Convolutional output shape (with 1x1 conv layer): "+str(tree._add_on(tree._net(xs)).shape))
        log.log_message("Prototypes shape: "+str(tree.prototype_layer.prototype_vectors.shape))

def analyse_leafs(tree: ProtoTree, epoch: int, k: int, leaf_labels: dict, threshold: float, log: Log):
    with torch.no_grad():
        if tree.depth<=4:
            log.log_message("class distributions of leaves:")
            for leaf in tree._root.leaves:
                if leaf._log_probabilities:
                    log.log_message(str(leaf.index)+", "+str(leaf._dist_params)+", "+str(torch.exp(leaf.distribution())))
                else:
                    log.log_message(str(leaf.index)+", "+str(leaf._dist_params)+", "+str(leaf.distribution()))


        leaf_labels[epoch] = []
        leafs_higher_than = []
        classes_covered = []
        
        for leaf in tree.leaves:
            label = torch.argmax(leaf._dist_params).item()
            
            if leaf._log_probabilities:
                value = torch.max(torch.exp(leaf.distribution())).item()
            else:
                value = torch.max(leaf.distribution()).item()
            if value >threshold:
                leafs_higher_than.append(leaf.index)
            leaf_labels[epoch].append((leaf.index, label))
            classes_covered.append(label)
        log.log_message("\nLeafs with max > %s: %s"%(threshold,len(leafs_higher_than)))

        class_without_leaf = 0
        for c in range(k):
            if c not in classes_covered:
                class_without_leaf +=1
        log.log_message("Classes without leaf: %s"%str(class_without_leaf))
        
        if len(leaf_labels.keys())>=2:
            changed_prev = 0
            changed_prev_higher = 0
            for pair in leaf_labels[epoch]:
                if pair not in leaf_labels[epoch-1]: #previous epoch
                    changed_prev +=1
                    if pair[0] in leafs_higher_than:
                        changed_prev_higher+=1
            log.log_message("Fraction changed pairs w.r.t previous epoch: %s"%str(changed_prev/float(tree.num_leaves)))
            if len(leafs_higher_than)> 0:
                log.log_message("Fraction changed leafs with max > threshold w.r.t previous epoch: %s"%str(changed_prev_higher/float(len(leafs_higher_than))))
    return leaf_labels


def analyse_ensemble(log, args, test_loader, device, trained_orig_trees, trained_pruned_trees, trained_pruned_projected_trees, orig_test_accuracies, pruned_test_accuracies, pruned_projected_test_accuracies, project_infos, infos_sample_max, infos_greedy, infos_fidelity):
    print("\nAnalysing and evaluating ensemble with %s trees of height %s..."%(len(trained_orig_trees), args.depth),flush=True)
    log.log_message("\n-----------------------------------------------------------------------------------------------------------------")
    log.log_message("\nAnalysing and evaluating ensemble with %s trees of height %s..."%(len(trained_orig_trees), args.depth))
    
    '''
    CALCULATE MEAN AND STANDARD DEVIATION BETWEEN RUNS
    '''
    log.log_message("Test accuracies of original individual trees: %s"%str(orig_test_accuracies))
    log.log_message("Mean and standard deviation of accuracies of original individual trees: \n"+ "mean="+str(np.mean(orig_test_accuracies))+", std="+str(np.std(orig_test_accuracies)))

    log.log_message("Test accuracies of pruned individual trees: %s"%str(pruned_test_accuracies))
    log.log_message("Mean and standard deviation of accuracies of pruned individual trees: \n"+ "mean="+str(np.mean(pruned_test_accuracies))+", std="+str(np.std(pruned_test_accuracies)))

    log.log_message("Test accuracies of pruned and projected individual trees: %s"%str(pruned_projected_test_accuracies))
    log.log_message("Mean and standard deviation of accuracies of pruned and projected individual trees:\n "+ "mean="+str(np.mean(pruned_projected_test_accuracies))+", std="+str(np.std(pruned_projected_test_accuracies)))

    '''
    CALCULATE MEAN NUMBER OF PROTOTYPES
    '''
    nums_prototypes = []
    for t in trained_pruned_trees:
        nums_prototypes.append(t.num_branches)
    log.log_message("Mean and standard deviation of number of prototypes in pruned trees:\n "+ "mean="+str(np.mean(nums_prototypes))+", std="+str(np.std(nums_prototypes)))

    '''
    CALCULATE MEAN DISTANCE TO NEAREST PROTOTYPE
    '''
    distances = []
    for i in range(len(trained_pruned_projected_trees)):
        info = project_infos[i]
        tree = trained_pruned_projected_trees[i]
        distances+=average_distance_nearest_image(info, tree, log,disable_log=True)
    log.log_message("Mean and standard deviation of distance from prototype to nearest training patch:\n "+ "mean="+str(np.mean(distances))+", std="+str(np.std(distances)))

    '''
    CALCULATE MEAN AND STANDARD DEVIATION BETWEEN RUNS WITH DETERMINISTIC ROUTING
    '''
    accuracies = []
    for info in infos_sample_max:
        accuracies.append(info['test_accuracy'])
    log.log_message("Mean and standard deviation of accuracies of pruned and projected individual trees with sample_max routing:\n "+ "mean="+str(np.mean(accuracies))+", std="+str(np.std(accuracies)))
    accuracies = []
    for info in infos_greedy:
        accuracies.append(info['test_accuracy'])
    log.log_message("Mean and standard deviation of accuracies of pruned and projected individual trees with greedy routing:\n "+ "mean="+str(np.mean(accuracies))+", std="+str(np.std(accuracies)))

    '''
    CALCULATE FIDELITY BETWEEN RUNS WITH DETERMINISTIC ROUTING
    '''
    fidelities_sample_max = []
    fidelities_greedy = []
    for info in infos_fidelity:
        fidelities_sample_max.append(info['distr_samplemax_fidelity'])
        fidelities_greedy.append(info['distr_greedy_fidelity'])
    log.log_message("Mean and standard deviation of fidelity of pruned and projected individual trees with sample_max routing:\n "+ "mean="+str(np.mean(fidelities_sample_max))+", std="+str(np.std(fidelities_sample_max)))
    log.log_message("Mean and standard deviation of fidelity of pruned and projected individual trees with greedy routing:\n "+ "mean="+str(np.mean(fidelities_greedy))+", std="+str(np.std(fidelities_greedy)))
    
    '''
    CALCULATE MEAN AND STANDARD DEVIATION OF PATH LENGTH WITH DETERMINISTIC ROUTING
    '''
    depths_sample_max = []
    depths_greedy = []
    for i in range(len(trained_pruned_projected_trees)):
        tree = trained_pruned_projected_trees[i]
        eval_info_sample_max = infos_sample_max[i]
        eval_info_greedy = infos_greedy[i]
        depths_sample_max+=get_avg_path_length(tree, eval_info_sample_max, log)
        depths_greedy+=get_avg_path_length(tree, eval_info_greedy, log)
    log.log_message("Mean and standard deviation of path length of pruned and projected individual trees with sample_max routing:\n "+ "mean="+str(np.mean(depths_sample_max))+", std="+str(np.std(depths_sample_max)))
    log.log_message("Tree with sample_max deterministic routing. Longest path has length %s, shortest path has length %s"%((np.max(depths_sample_max)), np.min(depths_sample_max)))
    log.log_message("Mean and standard deviation of path length of pruned and projected individual trees with greedy routing:\n "+ "mean="+str(np.mean(depths_greedy))+", std="+str(np.std(depths_greedy)))
    log.log_message("Tree with greedy deterministic routing. Longest path has length %s, shortest path has length %s"%((np.max(depths_greedy)), np.min(depths_greedy)))

    '''
    EVALUATE ENSEMBLE OF PRUNED AND PROJECTED TREES
    '''
    log.log_message("\nCalculating accuracy of tree ensemble with pruned and projected trees...")
    eval_ensemble(trained_pruned_projected_trees, test_loader, device, log, args, 'distributed')