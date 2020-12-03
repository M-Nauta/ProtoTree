
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import argparse
from subprocess import check_call
import math
from PIL import Image
from prototree.upsample import find_high_activation_crop, imsave_with_bbox
import torch

import torchvision
from torchvision.utils import save_image

from prototree.prototree import ProtoTree
from prototree.branch import Branch
from prototree.leaf import Leaf
from prototree.node import Node

def upsample_local(tree: ProtoTree,
                 sample: torch.Tensor,
                 sample_dir: str,
                 folder_name: str,
                 img_name: str,
                 decision_path: list,
                 args: argparse.Namespace):
    
    dir = os.path.join(os.path.join(os.path.join(args.log_dir, folder_name),img_name), args.dir_for_saving_images)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with torch.no_grad():
        _, distances_batch, _ = tree.forward_partial(sample)
        sim_map = torch.exp(-distances_batch[0,:,:,:]).cpu().numpy()
    for i, node in enumerate(decision_path[:-1]):
        decision_node_idx = node.index
        node_id = tree._out_map[node]
        img = Image.open(sample_dir)
        x_np = np.asarray(img)
        x_np = np.float32(x_np)/ 255
        if x_np.ndim == 2: #convert grayscale to RGB
            x_np = np.stack((x_np,)*3, axis=-1)
        
        img_size = x_np.shape[:2]
        similarity_map = sim_map[node_id]

        rescaled_sim_map = similarity_map - np.amin(similarity_map)
        rescaled_sim_map= rescaled_sim_map / np.amax(rescaled_sim_map)
        similarity_heatmap = cv2.applyColorMap(np.uint8(255*rescaled_sim_map), cv2.COLORMAP_JET)
        similarity_heatmap = np.float32(similarity_heatmap) / 255
        similarity_heatmap = similarity_heatmap[...,::-1]
        plt.imsave(fname=os.path.join(dir,'%s_heatmap_latent_similaritymap.png'%str(decision_node_idx)), arr=similarity_heatmap, vmin=0.0,vmax=1.0)

        upsampled_act_pattern = cv2.resize(similarity_map,
                                            dsize=(img_size[1], img_size[0]),
                                            interpolation=cv2.INTER_CUBIC)
        rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
        rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_original_img = 0.5 * x_np + 0.2 * heatmap
        plt.imsave(fname=os.path.join(dir,'%s_heatmap_original_image.png'%str(decision_node_idx)), arr=overlayed_original_img, vmin=0.0,vmax=1.0)

        # save the highly activated patch
        masked_similarity_map = np.ones(similarity_map.shape)
        masked_similarity_map[similarity_map < np.max(similarity_map)] = 0 #mask similarity map such that only the nearest patch z* is visualized
        
        upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                            dsize=(img_size[1], img_size[0]),
                                            interpolation=cv2.INTER_CUBIC)
        plt.imsave(fname=os.path.join(dir,'%s_masked_upsampled_heatmap.png'%str(decision_node_idx)), arr=upsampled_prototype_pattern, vmin=0.0,vmax=1.0) 
            
        high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, args.upsample_threshold)
        high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
                                            high_act_patch_indices[2]:high_act_patch_indices[3], :]
        plt.imsave(fname=os.path.join(dir,'%s_nearest_patch_of_image.png'%str(decision_node_idx)), arr=high_act_patch, vmin=0.0,vmax=1.0)

        # save the original image with bounding box showing high activation patch
        imsave_with_bbox(fname=os.path.join(dir,'%s_bounding_box_nearest_patch_of_image.png'%str(decision_node_idx)),
                            img_rgb=x_np,
                            bbox_height_start=high_act_patch_indices[0],
                            bbox_height_end=high_act_patch_indices[1],
                            bbox_width_start=high_act_patch_indices[2],
                            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

def gen_pred_vis(tree: ProtoTree,
                 sample: torch.Tensor,
                 sample_dir: str,
                 folder_name: str,
                 args: argparse.Namespace,
                 classes: tuple,
                 pred_kwargs: dict = None,
                 ):
    pred_kwargs = pred_kwargs or dict()  # TODO -- assert deterministic routing
    
    # Create dir to store visualization
    img_name = sample_dir.split('/')[-1].split(".")[-2]
    
    if not os.path.exists(os.path.join(args.log_dir, folder_name)):
        os.makedirs(os.path.join(args.log_dir, folder_name))
    destination_folder=os.path.join(os.path.join(args.log_dir, folder_name),img_name)
    
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    # Get references to where source files are stored
    upsample_path = os.path.join(os.path.join(args.log_dir,args.dir_for_saving_images),'pruned_and_projected')
    nodevis_path = os.path.join(args.log_dir,'pruned_and_projected/node_vis')
    local_upsample_path = os.path.join(destination_folder, args.dir_for_saving_images)

    # Get the model prediction
    with torch.no_grad():
        pred, pred_info = tree.forward(sample, sampling_strategy='greedy', **pred_kwargs)
        probs = pred_info['ps']
        label_ix = torch.argmax(pred, dim=1)[0].item()
        assert 'out_leaf_ix' in pred_info.keys()

    # Save input image
    sample_path = destination_folder + '/node_vis/sample.jpg'
    # save_image(sample, sample_path)
    Image.open(sample_dir).save(sample_path)

    # Save an image containing the model output
    output_path = destination_folder + '/node_vis/output.jpg'
    leaf_ix = pred_info['out_leaf_ix'][0]
    leaf = tree.nodes_by_index[leaf_ix]
    decision_path = tree.path_to(leaf)

    upsample_local(tree,sample,sample_dir,folder_name,img_name,decision_path,args)

    # Prediction graph is visualized using Graphviz
    # Build dot string
    s = 'digraph T {margin=0;rankdir=LR\n'
    # s += "subgraph {"
    s += 'node [shape=plaintext, label=""];\n'
    s += 'edge [penwidth="0.5"];\n'

    # Create a node for the sample image
    s += f'sample[image="{sample_path}"];\n'

    # Create nodes for all decisions/branches
    # Starting from the leaf
    for i, node in enumerate(decision_path[:-1]):
        node_ix = node.index
        prob = probs[node_ix].item()
        
        s += f'node_{i+1}[image="{upsample_path}/{node_ix}_nearest_patch_of_image.png" group="{"g"+str(i)}"];\n' 
        if prob > 0.5:
            s += f'node_{i+1}_original[image="{local_upsample_path}/{node_ix}_bounding_box_nearest_patch_of_image.png" imagescale=width group="{"g"+str(i)}"];\n'  
            label = "Present      \nSimilarity %.4f                   "%prob
            s += f'node_{i+1}->node_{i+1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        else:
            s += f'node_{i+1}_original[image="{sample_path}" group="{"g"+str(i)}"];\n'
            label = "Absent      \nSimilarity %.4f                   "%prob
            s += f'node_{i+1}->node_{i+1}_original [label="{label}" fontsize=10 fontname=Helvetica];\n'
        # s += f'node_{i+1}_original->node_{i+1} [label="{label}" fontsize=10 fontname=Helvetica];\n'
        
        s += f'node_{i+1}->node_{i+2};\n'
        s += "{rank = same; "f'node_{i+1}_original'+"; "+f'node_{i+1}'+"};"

    # Create a node for the model output
    s += f'node_{len(decision_path)}[imagepos="tc" imagescale=height image="{nodevis_path}/node_{leaf_ix}_vis.jpg" label="{classes[label_ix]}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'

    # Connect the input image to the first decision node
    s += 'sample->node_1;\n'


    s += '}\n'

    with open(os.path.join(destination_folder, 'predvis.dot'), 'w') as f:
        f.write(s)

    from_p = os.path.join(destination_folder, 'predvis.dot')
    to_pdf = os.path.join(destination_folder, 'predvis.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s' % (from_p, to_pdf), shell=True)


