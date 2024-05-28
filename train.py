import os
import math
import argparse
import multiprocessing as mp
import numpy as np

import torch

from configs import TrainingConfigs
from loss import CrossentropyLoss
from dataset import Dataset
from model import Model

parser = argparse.ArgumentParser(description="PyTorch Example")
parser.add_argument(
    "--batch-size",
    type=int,
    default=16,
    help="input batch size for training (default: 16)",
)
parser.add_argument(
    "--test-batch-size",
    type=int,
    default=16,
    help="input batch size for testing (default: 16)",
)
parser.add_argument(
    "--epoch",
    type=int,
    default=100,
    help="number of epochs to train (default: 100)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.001,
    help="learning rate (default: 0.001)",
)
parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
parser.add_argument(
    "--num-process",
    type=int,
    default=4,
    help="how many training processes to use (default: 4)",
)
parser.add_argument(
    "--cuda", action="store_true", default=False, help="enables CUDA training"
)
parser.add_argument(
    "--save-path",
    type=str,
    default="./weights",
    help="where to save the trained model weights",
)


def train(rank):

    torch.manual_seed(TrainingConfigs.argparse_args + rank)

    data_loader = torch.utils.data.DataLoader(TrainingConfigs.train_dataset)
    TrainingConfigs.model.train()

    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, data_loader)


def train_epoch(epoch, data_loader):

    pid = os.getpid()
    for batch_idx, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = torch.nn.functional.cross_entropy(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "{}\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    pid,
                    epoch,
                    batch_idx * len(data),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break


def test_epoch(model, device, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data.to(device))
            test_loss += F.nll_loss(
                output, target.to(device), reduction="sum"
            ).item()  # sum up batch loss
            pred = output.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target.to(device)).sum().item()

    test_loss /= len(data_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(data_loader.dataset),
            100.0 * correct / len(data_loader.dataset),
        )
    )


# if __name__ == "__main__":

#     # create losses (criterion in pytorch)
#     criterion_L1 = torch.nn.L1Loss()

#     # if running on GPU and we want to use cuda move model there

#     # create optimizers
#     optim = torch.optim.Adam(net.parameters(), lr=opt.lr)
#     ...

#     # load checkpoint if needed/ wanted
#     start_n_iter = 0
#     start_epoch = 0
#     if opt.resume:
#         ckpt = load_checkpoint(
#             opt.path_to_checkpoint
#         )  # custom method for loading last checkpoint
#         net.load_state_dict(ckpt["net"])
#         start_epoch = ckpt["epoch"]
#         start_n_iter = ckpt["n_iter"]
#         optim.load_state_dict(ckpt["optim"])
#         print("last checkpoint restored")
#         ...

#     # if we want to run experiment on multiple GPUs we move the models there
#     net = torch.nn.DataParallel(net)
#     ...

#     # typically we use tensorboardX to keep track of experiments
#     writer = SummaryWriter(...)

#     # now we start the main loop
#     n_iter = start_n_iter
#     for epoch in range(start_epoch, opt.epochs):
#         # set models to train mode
#         net.train()
#         ...

#         # use prefetch_generator and tqdm for iterating through data
#         pbar = tqdm(
#             enumerate(BackgroundGenerator(train_data_loader, ...)),
#             total=len(train_data_loader),
#         )
#         start_time = time.time()

#         # for loop going through dataset
#         for i, data in pbar:
#             # data preparation
#             img, label = data
#             if use_cuda:
#                 img = img.cuda()
#                 label = label.cuda()
#             ...

#             # It's very good practice to keep track of preparation time and computation time using tqdm to find any issues in your dataloader
#             prepare_time = start_time - time.time()

#             # forward and backward pass
#             optim.zero_grad()
#             ...
#             loss.backward()
#             optim.step()
#             ...

#             # udpate tensorboardX
#             writer.add_scalar(..., n_iter)
#             ...

#             # compute computation time and *compute_efficiency*
#             process_time = start_time - time.time() - prepare_time
#             pbar.set_description(
#                 "Compute efficiency: {:.2f}, epoch: {}/{}:".format(
#                     process_time / (process_time + prepare_time), epoch, opt.epochs
#                 )
#             )
#             start_time = time.time()

#         # maybe do a test pass every x epochs
#         if epoch % x == x - 1:
#             # bring models to evaluation mode
#             net.eval()
#             ...
#             # do some tests
#             pbar = tqdm(
#                 enumerate(BackgroundGenerator(test_data_loader, ...)),
#                 total=len(test_data_loader),
#             )
#             for i, data in pbar:
#                 ...

#             # save checkpoint if needed
#             ...


if __name__ == "__main__":

    args = parser.parse_args()

    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    mp.set_start_method("forkserver")  # Linux only
    torch.set_num_threads(int(mp.cpu_count() // args.num_processes))
    torch.backends.cudnn.benchmark = True

    model = Model(height=224, width=224, out_channels=16, num_class=3).to(device)
    model.share_memory()

    TrainingConfigs.argparse_args = args = parser.parse_args()
    TrainingConfigs.device = device
    TrainingConfigs.model = model
    TrainingConfigs.train_dataset = Dataset(data_root="./data/train", augment_data=True)
    TrainingConfigs.test_dataset = Dataset(data_root="./data/test", augment_data=False)
    TrainingConfigs.loss_function = CrossentropyLoss()

    processes = []

    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, TrainingConfigs))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
