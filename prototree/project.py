import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from prototree.prototree import ProtoTree
from util.log import Log

def project(tree: ProtoTree,
            project_loader: DataLoader,
            device,
            args: argparse.Namespace,
            log: Log,  
            log_prefix: str = 'log_projection',  # TODO
            progress_prefix: str = 'Projection'
            ) -> dict:
        
    log.log_message("\nProjecting prototypes to nearest training patch (without class restrictions)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                            total=len(project_loader),
                            desc=progress_prefix,
                            ncols=0
                            )

    
    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(project_loader))
        batch_size = xs.shape[0]
        for i, (xs, ys) in projection_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features_batch, distances_batch, out_map = tree.forward_partial(xs)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():

                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):

                    # Find the index of the latent patch that is closest to the prototype
                    min_distance = distances.min()
                    min_distance_ix = distances.argmin()
                    # Use the index to get the closest latent patch
                    closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                    # Check if the latent patch is closest for all data samples seen so far
                    if min_distance < global_min_proto_dist[j]:
                        global_min_proto_dist[j] = min_distance
                        global_min_patches[j] = closest_patch
                        global_min_info[j] = {
                            'input_image_ix': i * batch_size + batch_i,
                            'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                            'W': W,
                            'H': H,
                            'W1': W1,
                            'H1': H1,
                            'distance': min_distance.item(),
                            'nearest_input': torch.unsqueeze(xs[batch_i],0),
                            'node_ix': node.index,
                        }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del distances_batch
            del out_map
        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                                dim=0,
                                out=tree.prototype_layer.prototype_vectors)
        del projection

    return global_min_info, tree

def project_with_class_constraints(tree: ProtoTree,
                                    project_loader: DataLoader,
                                    device,
                                    args: argparse.Namespace,
                                    log: Log,  
                                    log_prefix: str = 'log_projection_with_constraints',  # TODO
                                    progress_prefix: str = 'Projection'
                                    ) -> dict:
        
    log.log_message("\nProjecting prototypes to nearest training patch (with class restrictions)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                            total=len(project_loader),
                            desc=progress_prefix,
                            ncols=0
                            )

    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(project_loader))
        batch_size = xs.shape[0]
        # For each internal node, collect the leaf labels in the subtree with this node as root. 
        # Only images from these classes can be used for projection.
        leaf_labels_subtree = dict()
        
        for branch, j in tree._out_map.items():
            leaf_labels_subtree[branch.index] = set()
            for leaf in branch.leaves:
                leaf_labels_subtree[branch.index].add(torch.argmax(leaf.distribution()).item())
        
        for i, (xs, ys) in projection_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features_batch, distances_batch, out_map = tree.forward_partial(xs)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():
                leaf_labels = leaf_labels_subtree[node.index]
                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
                    #Check if label of this image is in one of the leaves of the subtree
                    if ys[batch_i].item() in leaf_labels: 
                        # Find the index of the latent patch that is closest to the prototype
                        min_distance = distances.min()
                        min_distance_ix = distances.argmin()
                        # Use the index to get the closest latent patch
                        closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                        # Check if the latent patch is closest for all data samples seen so far
                        if min_distance < global_min_proto_dist[j]:
                            global_min_proto_dist[j] = min_distance
                            global_min_patches[j] = closest_patch
                            global_min_info[j] = {
                                'input_image_ix': i * batch_size + batch_i,
                                'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                                'W': W,
                                'H': H,
                                'W1': W1,
                                'H1': H1,
                                'distance': min_distance.item(),
                                'nearest_input': torch.unsqueeze(xs[batch_i],0),
                                'node_ix': node.index,
                            }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del distances_batch
            del out_map

        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                                dim=0, out=tree.prototype_layer.prototype_vectors)
        del projection

    return global_min_info, tree
