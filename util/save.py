import torch
import os
import argparse
from prototree.prototree import ProtoTree
from util.log import Log

def save_tree(tree: ProtoTree, optimizer, scheduler, epoch: int, log: Log, args: argparse.Namespace):
    tree.eval()
    # Save latest model
    tree.save(f'{log.checkpoint_dir}/latest')
    tree.save_state(f'{log.checkpoint_dir}/latest')
    torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/latest/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/latest/scheduler_state.pth')

    # Save model every 10 epochs
    if epoch == args.epochs or epoch%10==0:
        tree.save(f'{log.checkpoint_dir}/epoch_{epoch}')
        tree.save_state(f'{log.checkpoint_dir}/epoch_{epoch}')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/epoch_{epoch}/scheduler_state.pth')

def save_best_train_tree(tree: ProtoTree, optimizer, scheduler, best_train_acc: float, train_acc: float, log: Log):
    tree.eval()
    if train_acc > best_train_acc:
        best_train_acc = train_acc
        tree.save(f'{log.checkpoint_dir}/best_train_model')
        tree.save_state(f'{log.checkpoint_dir}/best_train_model')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/best_train_model/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/best_train_model/scheduler_state.pth')
    return best_train_acc

def save_best_test_tree(tree: ProtoTree, optimizer, scheduler, best_test_acc: float, test_acc: float, log: Log):
    tree.eval()
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        tree.save(f'{log.checkpoint_dir}/best_test_model')
        tree.save_state(f'{log.checkpoint_dir}/best_test_model')
        torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/best_test_model/optimizer_state.pth')
        torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/best_test_model/scheduler_state.pth')
    return best_test_acc

def save_tree_description(tree: ProtoTree, optimizer, scheduler, description: str, log: Log):
    tree.eval()
    # Save model with description
    tree.save(f'{log.checkpoint_dir}/'+description)
    tree.save_state(f'{log.checkpoint_dir}/'+description)
    torch.save(optimizer.state_dict(), f'{log.checkpoint_dir}/'+description+'/optimizer_state.pth')
    torch.save(scheduler.state_dict(), f'{log.checkpoint_dir}/'+description+'/scheduler_state.pth')
