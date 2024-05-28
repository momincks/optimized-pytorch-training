import argparse
import torch
import torch.utils
import torch.utils.data


class TrainingConfigs:
    device: str = None
    argparse_args: argparse.Namespace = None
    loss_function: torch.nn.Module = None
    optimizer: torch.nn.Module = None
    model: torch.nn.Module = None
    train_dataset: torch.utils.data.Dataset = None
    # train_dataloader: torch.utils.data.DataLoader = None
    val_dataset: torch.utils.data.Dataset = None
    # val_dataloader: torch.utils.data.DataLoader = None
    test_dataset: torch.utils.data.Dataset = None
    # test_dataloader: torch.utils.data.DataLoader = None
