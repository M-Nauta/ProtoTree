import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader

from prototree.prototree import ProtoTree
from util.log import Log


@torch.no_grad()
def eval(tree: ProtoTree,
        test_loader: DataLoader,
        epoch,
        device,
        log: Log = None,  
        sampling_strategy: str = 'distributed',
        log_prefix: str = 'log_eval_epochs', 
        progress_prefix: str = 'Eval Epoch'
        ) -> dict:
    tree = tree.to(device)

    # Keep an info dict about the procedure
    info = dict()
    if sampling_strategy != 'distributed':
        info['out_leaf_ix'] = []
    # Build a confusion matrix
    cm = np.zeros((tree._num_classes, tree._num_classes), dtype=int)

    # Make sure the model is in evaluation mode
    tree.eval()

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix+' %s'%epoch,
                        ncols=0)

    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data
        out, test_info = tree.forward(xs, sampling_strategy)
        ys_pred = torch.argmax(out, dim=1)

        # Update the confusion matrix
        cm_batch = np.zeros((tree._num_classes, tree._num_classes), dtype=int)
        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true][y_pred] += 1
            cm_batch[y_true][y_pred] += 1
        acc = acc_from_cm(cm_batch)
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}], Acc: {acc:.3f}'
        )

        # keep list of leaf indices where test sample ends up when deterministic routing is used.
        if sampling_strategy != 'distributed':
            info['out_leaf_ix'] += test_info['out_leaf_ix']
        del out
        del ys_pred
        del test_info

    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    log.log_message("\nEpoch %s - Test accuracy with %s routing: "%(epoch, sampling_strategy)+str(info['test_accuracy']))
    return info

@torch.no_grad()
def eval_fidelity(tree: ProtoTree,
        test_loader: DataLoader,
        device,
        log: Log = None,  
        progress_prefix: str = 'Fidelity'
        ) -> dict:
    tree = tree.to(device)

    # Keep an info dict about the procedure
    info = dict()

    # Make sure the model is in evaluation mode
    tree.eval()
    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix,
                        ncols=0)

    distr_samplemax_fidelity = 0
    distr_greedy_fidelity = 0
    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data, with 3 types of routing
        out_distr, _ = tree.forward(xs, 'distributed')
        ys_pred_distr = torch.argmax(out_distr, dim=1)

        out_samplemax, _ = tree.forward(xs, 'sample_max')
        ys_pred_samplemax = torch.argmax(out_samplemax, dim=1)

        out_greedy, _ = tree.forward(xs, 'greedy')
        ys_pred_greedy = torch.argmax(out_greedy, dim=1)
        
        # Calculate fidelity
        distr_samplemax_fidelity += torch.sum(torch.eq(ys_pred_samplemax, ys_pred_distr)).item()
        distr_greedy_fidelity += torch.sum(torch.eq(ys_pred_greedy, ys_pred_distr)).item()
        # Update the progress bar
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}]'
        )
        del out_distr
        del out_samplemax
        del out_greedy

    distr_samplemax_fidelity = distr_samplemax_fidelity/float(len(test_loader.dataset))
    distr_greedy_fidelity = distr_greedy_fidelity/float(len(test_loader.dataset))
    info['distr_samplemax_fidelity'] = distr_samplemax_fidelity
    info['distr_greedy_fidelity'] = distr_greedy_fidelity
    log.log_message("Fidelity between standard distributed routing and sample_max routing: "+str(distr_samplemax_fidelity))
    log.log_message("Fidelity between standard distributed routing and greedy routing: "+str(distr_greedy_fidelity))
    return info

@torch.no_grad()
def eval_ensemble(trees: list, test_loader: DataLoader, device, log: Log, args: argparse.Namespace, sampling_strategy: str = 'distributed', progress_prefix: str = 'Eval Ensemble'):
    # Keep an info dict about the procedure
    info = dict()
    # Build a confusion matrix
    cm = np.zeros((trees[0]._num_classes, trees[0]._num_classes), dtype=int)    

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc=progress_prefix,
                        ncols=0)

    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)
        outs = []
        for tree in trees:
            # Make sure the model is in evaluation mode
            tree.eval()
            tree = tree.to(device)
            # Use the model to classify this batch of input data
            out, _ = tree.forward(xs, sampling_strategy)
            outs.append(out)
            del out
        stacked = torch.stack(outs, dim=0)
        ys_pred = torch.argmax(torch.mean(stacked, dim=0), dim=1)
        
        for y_pred, y_true in zip(ys_pred, ys):
            cm[y_true][y_pred] += 1
            
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}]'
        )
        del outs
            
    info['confusion_matrix'] = cm
    info['test_accuracy'] = acc_from_cm(cm)
    log.log_message("Ensemble accuracy with %s routing: %s"%(sampling_strategy, str(info['test_accuracy'])))
    return info

def acc_from_cm(cm: np.ndarray) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    correct = 0
    for i in range(len(cm)):
        correct += cm[i, i]

    total = np.sum(cm)
    if total == 0:
        return 1
    else:
        return correct / total
