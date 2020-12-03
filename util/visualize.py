import os
import subprocess
import numpy as np
import copy
import argparse
from subprocess import check_call
from PIL import Image
import torch
import math
from prototree.prototree import ProtoTree
from prototree.branch import Branch
from prototree.leaf import Leaf
from prototree.node import Node


def gen_vis(tree: ProtoTree, folder_name: str, args: argparse.Namespace, classes:tuple):
    destination_folder=os.path.join(args.log_dir,folder_name)
    upsample_dir = os.path.join(os.path.join(args.log_dir, args.dir_for_saving_images), folder_name)
    if not os.path.isdir(destination_folder):
        os.mkdir(destination_folder)
    if not os.path.isdir(destination_folder + '/node_vis'):
        os.mkdir(destination_folder + '/node_vis')

    with torch.no_grad():
        s = 'digraph T {margin=0;ranksep=".03";nodesep="0.05";splines="false";\n'
        s += 'node [shape=rect, label=""];\n'
        s += _gen_dot_nodes(tree._root, destination_folder, upsample_dir, classes)
        s += _gen_dot_edges(tree._root, classes)[0]
        s += '}\n'

    with open(os.path.join(destination_folder,'treevis.dot'), 'w') as f:
        f.write(s)
   
    from_p = os.path.join(destination_folder,'treevis.dot')
    to_pdf = os.path.join(destination_folder,'treevis.pdf')
    check_call('dot -Tpdf -Gmargin=0 %s -o %s'%(from_p, to_pdf), shell=True)

def _node_vis(node: Node, upsample_dir: str):
    if isinstance(node, Leaf):
        return _leaf_vis(node)
    if isinstance(node, Branch):
        return _branch_vis(node, upsample_dir)


def _leaf_vis(node: Leaf):
    if node._log_probabilities:
        ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
    else:
        ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
    
    ws = np.ones(ws.shape) - ws
    ws *= 255

    height = 24

    if ws.shape[0] < 36:
        img_size = 36
    else:
        img_size = ws.shape[0]
    scaler = math.ceil(img_size/ws.shape[0])

    img = Image.new('F', (ws.shape[0]*scaler, height))
    pixels = img.load()

    for i in range(scaler*ws.shape[0]):
        for j in range(height-10):
            pixels[i,j]=ws[int(i/scaler)]
        for j in range(height-10,height-9):
            pixels[i,j]=0 #set bottom line of leaf distribution black
        for j in range(height-9,height):
            pixels[i,j]=255 #set bottom part of node white such that class label is readable

    if scaler*ws.shape[0]>100:
        img=img.resize((100,height))
    return img


def _branch_vis(node: Branch, upsample_dir: str):
    branch_id = node.index
    
    img = Image.open(os.path.join(upsample_dir, '%s_nearest_patch_of_image.png'%branch_id))
    bb = Image.open(os.path.join(upsample_dir, '%s_bounding_box_nearest_patch_of_image.png'%branch_id))
    map = Image.open(os.path.join(upsample_dir, '%s_heatmap_original_image.png'%branch_id))
    w, h = img.size
    wbb, hbb = bb.size
    
    if wbb < 100 and hbb < 100:
        cs = wbb, hbb
    else:
        cs = 100/wbb, 100/hbb
        min_cs = min(cs)
        bb = bb.resize(size=(int(min_cs * wbb), int(min_cs * hbb)))
        wbb, hbb = bb.size

    if w < 100 and h < 100:
        cs = w, h
    else:
        cs = 100/w, 100/h
        min_cs = min(cs)
        img = img.resize(size=(int(min_cs * w), int(min_cs * h)))
        w, h = img.size

    between = 4
    total_w = w+wbb + between
    total_h = max(h, hbb)
    

    together = Image.new(img.mode, (total_w, total_h), color=(255,255,255))
    together.paste(img, (0, 0))
    together.paste(bb, (w+between, 0))

    return together


def _gen_dot_nodes(node: Node, destination_folder: str, upsample_dir: str, classes:tuple):
    img = _node_vis(node, upsample_dir).convert('RGB')
    if isinstance(node, Leaf):
        if node._log_probabilities:
            ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        str_targets = ','.join(str(t) for t in class_targets) if len(class_targets) > 0 else ""
        str_targets = str_targets.replace('_', ' ')
    filename = '{}/node_vis/node_{}_vis.jpg'.format(destination_folder, node.index)
    img.save(filename)
    if isinstance(node, Leaf):
        s = '{}[imagepos="tc" imagescale=height image="{}" label="{}" labelloc=b fontsize=10 penwidth=0 fontname=Helvetica];\n'.format(node.index, filename, str_targets)
    else:
        s = '{}[image="{}" xlabel="{}" fontsize=6 labelfontcolor=gray50 fontname=Helvetica];\n'.format(node.index, filename, node.index)
    if isinstance(node, Branch):
        return s\
               + _gen_dot_nodes(node.l, destination_folder, upsample_dir, classes)\
               + _gen_dot_nodes(node.r, destination_folder, upsample_dir, classes)
    if isinstance(node, Leaf):
        return s


def _gen_dot_edges(node: Node, classes:tuple):
    if isinstance(node, Branch):
        edge_l, targets_l = _gen_dot_edges(node.l, classes)
        edge_r, targets_r = _gen_dot_edges(node.r, classes)
        str_targets_l = ','.join(str(t) for t in targets_l) if len(targets_l) > 0 else ""
        str_targets_r = ','.join(str(t) for t in targets_r) if len(targets_r) > 0 else ""
        s = '{} -> {} [label="Absent" fontsize=10 tailport="s" headport="n" fontname=Helvetica];\n {} -> {} [label="Present" fontsize=10 tailport="s" headport="n" fontname=Helvetica];\n'.format(node.index, node.l.index, 
                                                                       node.index, node.r.index)
        return s + edge_l + edge_r, sorted(list(set(targets_l + targets_r)))
    if isinstance(node, Leaf):
        if node._log_probabilities:
            ws = copy.deepcopy(torch.exp(node.distribution()).cpu().detach().numpy())
        else:
            ws = copy.deepcopy(node.distribution().cpu().detach().numpy())
        argmax = np.argmax(ws)
        targets = [argmax] if argmax.shape == () else argmax.tolist()
        class_targets = copy.deepcopy(targets)
        for i in range(len(targets)):
            t = targets[i]
            class_targets[i] = classes[t]
        return '', class_targets

