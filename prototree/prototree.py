
import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn

from prototree.branch import Branch
from prototree.leaf import Leaf
from prototree.node import Node
from util.func import min_pool2d

from util.l2conv import L2Conv2D

class ProtoTree(nn.Module):

    ARGUMENTS = ['depth', 'num_features', 'W1', 'H1', 'log_probabilities']

    SAMPLING_STRATEGIES = ['distributed', 'sample_max', 'greedy']

    def __init__(self,
                 num_classes: int,
                 feature_net: torch.nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module = nn.Identity(),
                 ):
        super().__init__()
        assert args.depth > 0
        assert num_classes > 0

        self._num_classes = num_classes

        # Build the tree
        self._root = self._init_tree(num_classes, args)

        self.num_features = args.num_features
        self.num_prototypes = self.num_branches
        self.prototype_shape = (args.W1, args.H1, args.num_features)
        
        # Keep a dict that stores a reference to each node's parent
        # Key: node -> Value: the node's parent
        # The root of the tree is mapped to None
        self._parents = dict()
        self._set_parents()  # Traverse the tree to build the self._parents dict

        # Set the feature network
        self._net = feature_net
        self._add_on = add_on_layers

        # Flag that indicates whether probabilities or log probabilities are computed
        self._log_probabilities = args.log_probabilities

        # Flag that indicates whether a normalization factor should be used instead of softmax. 
        self._kontschieder_normalization = args.kontschieder_normalization
        self._kontschieder_train = args.kontschieder_train
        # Map each decision node to an output of the feature net
        self._out_map = {n: i for i, n in zip(range(2 ** (args.depth) - 1), self.branches)}

        self.prototype_layer = L2Conv2D(self.num_prototypes,
                                        self.num_features,
                                        args.W1,
                                        args.H1)

    @property
    def root(self) -> Node:
        return self._root

    @property
    def leaves_require_grad(self) -> bool:
        return any([leaf.requires_grad for leaf in self.leaves])

    @leaves_require_grad.setter
    def leaves_require_grad(self, val: bool):
        for leaf in self.leaves:
            leaf.requires_grad = val

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    def forward(self,
                xs: torch.Tensor,
                sampling_strategy: str = SAMPLING_STRATEGIES[0],  # `distributed` by default
                **kwargs,
                ) -> tuple:
        assert sampling_strategy in ProtoTree.SAMPLING_STRATEGIES

        '''
            PERFORM A FORWARD PASS THROUGH THE FEATURE NET
        '''

        # Perform a forward pass with the conv net
        features = self._net(xs)
        features = self._add_on(features)
        bs, D, W, H = features.shape

        '''
            COMPUTE THE PROTOTYPE SIMILARITIES GIVEN THE COMPUTED FEATURES
        '''

        # Use the features to compute the distances from the prototypes
        distances = self.prototype_layer(features)  # Shape: (batch_size, num_prototypes, W, H)

        # Perform global min pooling to see the minimal distance for each prototype to any patch of the input image
        min_distances = min_pool2d(distances, kernel_size=(W, H))
        min_distances = min_distances.view(bs, self.num_prototypes)

        if not self._log_probabilities:
            similarities = torch.exp(-min_distances)
        else:
            # Omit the exp since we require log probabilities
            similarities = -min_distances

        # Add the conv net output to the kwargs dict to be passed to the decision nodes in the tree
        # Split (or chunk) the conv net output tensor of shape (batch_size, num_decision_nodes) into individual tensors
        # of shape (batch_size, 1) containing the logits that are relevant to single decision nodes
        kwargs['conv_net_output'] = similarities.chunk(similarities.size(1), dim=1)
        # Add the mapping of decision nodes to conv net outputs to the kwargs dict to be passed to the decision nodes in
        # the tree
        kwargs['out_map'] = dict(self._out_map)  # Use a copy of self._out_map, as the original should not be modified

        '''
            PERFORM A FORWARD PASS THROUGH THE TREE GIVEN THE COMPUTED SIMILARITIES
        '''

        # Perform a forward pass through the tree
        out, attr = self._root.forward(xs, **kwargs)

        info = dict()
        # Store the probability of arriving at all nodes in the decision tree
        info['pa_tensor'] = {n.index: attr[n, 'pa'].unsqueeze(1) for n in self.nodes}
        # Store the output probabilities of all decision nodes in the tree
        info['ps'] = {n.index: attr[n, 'ps'].unsqueeze(1) for n in self.branches}

        # Generate the output based on the chosen sampling strategy
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[0]:  # Distributed
            return out, info
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[1]:  # Sample max
            # Get the batch size
            batch_size = xs.size(0)
            # Get an ordering of all leaves in the tree
            leaves = list(self.leaves)
            # Obtain path probabilities of arriving at each leaf
            pas = [attr[l, 'pa'].view(batch_size, 1) for l in leaves]  # All shaped (bs, 1)
            # Obtain output distributions of each leaf
            dss = [attr[l, 'ds'].view(batch_size, 1, self._num_classes) for l in leaves]  # All shaped (bs, 1, k)
            # Prepare data for selection of most probable distributions
            # Let L denote the number of leaves in this tree
            pas = torch.cat(tuple(pas), dim=1)  # shape: (bs, L)
            dss = torch.cat(tuple(dss), dim=1)  # shape: (bs, L, k)
            # Select indices (in the 'leaves' variable) of leaves with highest path probability
            ix = torch.argmax(pas, dim=1).long()  # shape: (bs,)
            # Select distributions of leafs with highest path probability
            dists = []
            for j, i in zip(range(dss.shape[0]), ix):
                dists += [dss[j][i].view(1, -1)]  # All shaped (1, k)
            dists = torch.cat(tuple(dists), dim=0)  # shape: (bs, k)

            # Store the indices of the leaves with the highest path probability
            info['out_leaf_ix'] = [leaves[i.item()].index for i in ix]

            return dists, info
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[2]:  # Greedy
            # At every decision node, the child with highest probability will be chosen
            batch_size = xs.size(0)
            # Set the threshold for when either child is more likely
            threshold = 0.5 if not self._log_probabilities else np.log(0.5)
            # Keep track of the routes taken for each of the items in the batch
            routing = [[] for _ in range(batch_size)]
            # Traverse the tree for all items
            # Keep track of all nodes encountered
            for i in range(batch_size):
                node = self._root
                while node in self.branches:
                    routing[i] += [node]
                    if attr[node, 'ps'][i].item() > threshold:
                        node = node.r
                    else:
                        node = node.l
                routing[i] += [node]

            # Obtain output distributions of each leaf
            # Each selected leaf is at the end of a path stored in the `routing` variable
            dists = [attr[path[-1], 'ds'][0] for path in routing]
            # Concatenate the dists in a new batch dimension
            dists = torch.cat([dist.unsqueeze(0) for dist in dists], dim=0).to(device=xs.device)

            # Store info
            info['out_leaf_ix'] = [path[-1].index for path in routing]

            return dists, info
        raise Exception('Sampling strategy not recognized!')

    def forward_partial(self, xs: torch.Tensor) -> tuple:

        # Perform a forward pass with the conv net
        features = self._net(xs)
        features = self._add_on(features)

        # Use the features to compute the distances from the prototypes
        distances = self.prototype_layer(features)  # Shape: (batch_size, num_prototypes, W, H)

        return features, distances, dict(self._out_map)

    @property
    def depth(self) -> int:
        d = lambda node: 1 if isinstance(node, Leaf) else 1 + max(d(node.l), d(node.r))
        return d(self._root)

    @property
    def size(self) -> int:
        return self._root.size

    @property
    def nodes(self) -> set:
        return self._root.nodes

    @property
    def nodes_by_index(self) -> dict:
        return self._root.nodes_by_index

    @property
    def node_depths(self) -> dict:

        def _assign_depths(node, d):
            if isinstance(node, Leaf):
                return {node: d}
            if isinstance(node, Branch):
                return {node: d, **_assign_depths(node.r, d + 1), **_assign_depths(node.l, d + 1)}

        return _assign_depths(self._root, 0)

    @property
    def branches(self) -> set:
        return self._root.branches

    @property
    def leaves(self) -> set:
        return self._root.leaves

    @property
    def num_branches(self) -> int:
        return self._root.num_branches

    @property
    def num_leaves(self) -> int:
        return self._root.num_leaves

    def save(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model_state.pth', 'wb') as f:
            torch.save(self.state_dict(), f)
        # Save the out_map of the model to the target directory
        with open(directory_path + '/tree.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        
    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + '/model.pth')      
       
    def _init_tree(self,
                   num_classes,
                   args: argparse.Namespace) -> Node:

        def _init_tree_recursive(i: int, d: int) -> Node:  # Recursively build the tree
            if d == args.depth:
                return Leaf(i,
                            num_classes,
                            args
                            )
            else:
                left = _init_tree_recursive(i + 1, d + 1)
                return Branch(i,
                              left,
                              _init_tree_recursive(i + left.size + 1, d + 1),
                              args,
                              )

        return _init_tree_recursive(0, 0)

    def _set_parents(self) -> None:
        self._parents.clear()
        self._parents[self._root] = None

        def _set_parents_recursively(node: Node):
            if isinstance(node, Branch):
                self._parents[node.r] = node
                self._parents[node.l] = node
                _set_parents_recursively(node.r)
                _set_parents_recursively(node.l)
                return
            if isinstance(node, Leaf):
                return  # Nothing to do here!
            raise Exception('Unrecognized node type!')

        # Set all parents by traversing the tree starting from the root
        _set_parents_recursively(self._root)

    def path_to(self, node: Node):
        assert node in self.leaves or node in self.branches
        path = [node]
        while isinstance(self._parents[node], Node):
            node = self._parents[node]
            path = [node] + path
        return path


