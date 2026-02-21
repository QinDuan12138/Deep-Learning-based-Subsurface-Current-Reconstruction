import argparse
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from model import SwinUnet as ViT_seg
from utils import plot_loss_curves
from config import get_config


def train_main(config):
    """
    Main function containing all training logic, called by each DDP process.
    """
    # ===== 1. Initialize DDP =====
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # ===== 2. Load datasets =====
    from datasets import MyDataset
    db_train = MyDataset(
        x_dir="/data/AI_eddy_current/data_daily_new_processed/train/X",
        y_dir="/data/AI_eddy_current/data_daily_new_processed/train/Y",
        start_date="19930101", end_date="20171231"
    )
    db_val = MyDataset(
        x_dir="/root/shared-nvme/AI_eddy_current/data_daily_processed/train/X",
        y_dir="/root/shared-nvme/AI_eddy_current/data_daily_processed/train/Y",
        start_date="20180101", end_date="20191231"
    )

    if rank == 0:
        print(f"Using {world_size} GPUs for training.")
        print(f"The length of train set is: {len(db_train)}")
        print(f"The length of valid set is: {len(db_val)}")

    train_sampler = DistributedSampler(db_train, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(db_val, num_replicas=world_size, rank=rank, shuffle=False)

    # Fix: use per-GPU batch size
    per_gpu_batch_size = config.DATA.BATCH_SIZE // world_size

    train_loader = DataLoader(
        db_train,
        batch_size=per_gpu_batch_size,
        sampler=train_sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        db_val,
        batch_size=per_gpu_batch_size,
        sampler=val_sampler,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=True
    )

    if rank == 0:
        print(f"The total batch size is: {config.DATA.BATCH_SIZE}")
        print(f"The batch size per gpu is: {per_gpu_batch_size}")

    # ===== 3. Model =====
    model = ViT_seg(config, img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # ===== 4. Optimizer & loss function =====
    base_lr = config.TRAIN.BASE_LR
    if rank == 0:
        print(f"The scaled base learning rate is: {base_lr}")
    mse_loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)

    max_iterations = config.TRAIN.EPOCHS * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iterations, eta_min=1e-6)

    iter_num = 0
    max_epoch = config.TRAIN.EPOCHS
    best_loss = float("inf")
    eval_interval = config.MISC.EVAL_INTERVAL

    train_losses = []
    val_losses = []

    # ===== 5. Early stopping =====
    patience = 20
    patience_counter = 0

    # ===== 6. Training loop =====
    for epoch_num in range(max_epoch):
        train_sampler.set_epoch(epoch_num)

        model.train()
        batch_mse_loss = 0

        # tqdm is only shown on rank 0
        train_loader_tqdm = tqdm(train_loader, desc=f"Train: {epoch_num}", leave=False) if rank == 0 else train_loader

        for i_batch, (image_batch, label_batch) in enumerate(train_loader_tqdm):
            image_batch, label_batch = image_batch.to(local_rank), label_batch.to(local_rank)

            outputs = model(image_batch)
            loss = mse_loss(outputs, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iter_num += 1

            # Only print on rank 0
            if rank == 0 and iter_num % 20 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"Iteration: {iter_num}/{max_iterations}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

            batch_mse_loss += loss.item()

        # Aggregate training loss across all processes
        batch_mse_loss_tensor = torch.tensor(batch_mse_loss, device=local_rank)
        dist.all_reduce(batch_mse_loss_tensor, op=dist.ReduceOp.SUM)
        batch_mse_loss = batch_mse_loss_tensor.item() / (len(train_loader) * world_size)

        if rank == 0:
            print(f"Epoch {epoch_num} - Train MSE Loss: {batch_mse_loss:.4f}")
            train_losses.append(batch_mse_loss)

        # ===== 7. Validation (all processes) =====
        stop_training = torch.tensor(0, device=local_rank)
        if (epoch_num + 1) % eval_interval == 0:
            model.eval()
            val_mse_loss = 0
            val_total_batches = 0
            with torch.no_grad():
                val_loader_tqdm = tqdm(val_loader, desc=f"Val: {epoch_num}", leave=False) if rank == 0 else val_loader
                for i_batch, (image_batch, label_batch) in enumerate(val_loader_tqdm):
                    image_batch, label_batch = image_batch.to(local_rank), label_batch.to(local_rank)
                    outputs = model(image_batch)
                    loss = mse_loss(outputs, label_batch)
                    val_mse_loss += loss.item()
                    val_total_batches += 1

            # Aggregate validation loss across all processes
            val_mse_loss_tensor = torch.tensor(val_mse_loss, device=local_rank)
            dist.all_reduce(val_mse_loss_tensor, op=dist.ReduceOp.SUM)

            val_total_batches_tensor = torch.tensor(val_total_batches, device=local_rank)
            dist.all_reduce(val_total_batches_tensor, op=dist.ReduceOp.SUM)

            val_mse_loss = val_mse_loss_tensor.item() / val_total_batches_tensor.item()

            if rank == 0:
                print(f"Epoch {epoch_num} - Val MSE Loss: {val_mse_loss:.4f}")
                val_losses.append(val_mse_loss)

                # ===== Save model & early stop check (only on rank=0) =====
                if val_mse_loss < best_loss:
                    save_mode_path = os.path.join(config.MISC.OUTPUT, "best_model.pth")
                    torch.save(model.module.state_dict(), save_mode_path)
                    best_loss = val_mse_loss
                    patience_counter = 0
                    print(f"New best model saved at {save_mode_path} with loss {best_loss:.4f}")
                else:
                    patience_counter += 1
                    save_mode_path = os.path.join(config.MISC.OUTPUT, "last_model.pth")
                    torch.save(model.module.state_dict(), save_mode_path)
                    print(f"Model saved at {save_mode_path} (no improvement). Patience {patience_counter}/{patience}")

                np.save(os.path.join(config.MISC.OUTPUT, "train_losses.npy"), np.array(train_losses))
                np.save(os.path.join(config.MISC.OUTPUT, "val_losses.npy"), np.array(val_losses))
                plot_loss_curves(train_losses, val_losses, save_path=os.path.join(config.MISC.OUTPUT, "loss_curve.png"))

                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch_num + 1}. Best val loss: {best_loss:.4f}")
                    stop_training.fill_(1)

        # ===== Broadcast early stop signal so all processes exit together =====
        dist.broadcast(stop_training, src=0)
        if stop_training.item() == 1:
            break

    dist.destroy_process_group()
    return "Training Finished!"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str,
                        default="/data/AI_eddy_current/script/DQ/SwinUnet_new_website", help="root dir for data")
    parser.add_argument("--list_dir", type=str,
                        default="./lists/lists_Synapse", help="list dir")
    parser.add_argument("--num_classes", type=int,
                        default=44, help="output channel of network")
    parser.add_argument("--output_dir", type=str,
                        default="../data/model_weights", help="output dir")
    parser.add_argument("--max_epochs", type=int,
                        default=250, help="maximum epoch number to train")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="total batch_size across all gpus")
    parser.add_argument("--n_gpu", type=int, default=4, help="total gpu")
    parser.add_argument("--deterministic", type=int, default=1,
                        help="whether use deterministic training")
    parser.add_argument("--base_lr", type=float, default=0.001,
                        help="segmentation network learning rate")
    parser.add_argument("--img_size", type=int,
                        default=512, help="input patch size of network input")
    parser.add_argument("--seed", type=int,
                        default=1234, help="random seed")
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.",
                        default=None, nargs="+")

    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument("--accumulation-steps", type=int, help="gradient accumulation steps")
    parser.add_argument("--use-checkpoint", action="store_true",
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument("--amp-opt-level", type=str, default="O1", choices=["O0", "O1", "O2"],
                        help="mixed precision opt level, if O0, no amp is used")
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument("--throughput", action="store_true", help="Test throughput only")
    parser.add_argument("--n_class", default=44, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--eval_interval", default=1, type=int)

    args = parser.parse_args()
    config = get_config(args)

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_main(config)


if __name__ == "__main__":
    main()
    
    
# torchrun --nproc_per_node=4 train_ddp.py