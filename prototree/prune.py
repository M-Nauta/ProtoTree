from prototree.prototree import ProtoTree
from prototree.branch import Branch
from prototree.leaf import Leaf
from prototree.node import Node
from util.log import Log
from copy import deepcopy
import torch

# Collects the nodes 
def nodes_to_prune_based_on_leaf_dists_threshold(tree: ProtoTree, threshold: float) -> list:
    to_prune_incl_possible_children = []
    for node in tree.nodes:
        if has_max_prob_lower_threshold(node, threshold):
            #prune everything below incl this node
            to_prune_incl_possible_children.append(node.index)
    return to_prune_incl_possible_children

# Returns True when all the node's children have a max leaf value < threshold
def has_max_prob_lower_threshold(node: Node, threshold: float):
    if isinstance(node, Branch):
        for leaf in node.leaves:
            if leaf._log_probabilities:
                if torch.max(torch.exp(leaf.distribution())).item() > threshold: 
                    return False
            else:
                if torch.max(leaf.distribution()).item() > threshold: 
                    return False
    elif isinstance(node, Leaf):
        if node._log_probabilities:
            if torch.max(torch.exp(node.distribution())).item() > threshold: 
                return False
        else:
            if torch.max(node.distribution()).item() > threshold: 
                return False
    else:
        raise Exception('This node type should not be possible. A tree has branches and leaves.')
    return True


# Prune tree
def prune(tree: ProtoTree, pruning_threshold_leaves: float, log: Log) -> list:
    log.log_message("\nPruning...")
    log.log_message("Before pruning: %s branches and %s leaves"%(tree.num_branches,tree.num_leaves))
    num_prototypes_before = tree.num_branches
    node_idxs_to_prune = nodes_to_prune_based_on_leaf_dists_threshold(tree, pruning_threshold_leaves)
    to_prune = deepcopy(node_idxs_to_prune)
    # remove children from prune_list of nodes that would already be pruned
    for node_idx in node_idxs_to_prune:
        if isinstance(tree.nodes_by_index[node_idx], Branch):
            if node_idx>0: #parent cannot be root since root would then be removed
                for child in tree.nodes_by_index[node_idx].nodes:
                    if child.index in to_prune and child.index != node_idx:
                        to_prune.remove(child.index)
    
    for node_idx in to_prune:
        node = tree.nodes_by_index[node_idx]
        parent = tree._parents[node]
        if parent.index>0: #parent cannot be root since root would then be removed
            if node == parent.l:
                if parent == tree._parents[parent].l:
                    #make right child of parent the left child of parent of parent
                    tree._parents[parent.r] = tree._parents[parent]
                    tree._parents[parent].l = parent.r
                elif parent == tree._parents[parent].r:
                    #make right child of parent the right child of parent of parent
                    tree._parents[parent.r] = tree._parents[parent]
                    tree._parents[parent].r = parent.r
                else:
                    raise Exception('Pruning went wrong, this should not be possible')

            elif node == parent.r:
                if parent == tree._parents[parent].l:
                    #make left child or parent the left child of parent of parent
                    tree._parents[parent.l] = tree._parents[parent]
                    tree._parents[parent].l = parent.l
                elif parent == tree._parents[parent].r:
                    #make left child of parent the right child of parent of parent
                    tree._parents[parent.l] = tree._parents[parent]
                    tree._parents[parent].r = parent.l
                else:
                    raise Exception('Pruning went wrong, this should not be possible')
            else:
                raise Exception('Pruning went wrong, this should not be possible')

    log.log_message("After pruning: %s branches and %s leaves"%(tree.num_branches,tree.num_leaves))
    log.log_message("Fraction of prototypes pruned: %s"%((num_prototypes_before-tree.num_branches)/float(num_prototypes_before))+'\n')
