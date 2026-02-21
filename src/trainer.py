import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from utils import plot_loss_curves

def trainer_AIEC(args, model, snapshot_path):
    from datasets import MyDataset
    base_lr = args.base_lr
    print(base_lr)
    batch_size = args.batch_size * args.n_gpu
    print(f'batch size{batch_size}')
    # Use 25 years as the training set
    db_train = MyDataset(x_dir= "/root/shared-nvme/AI_eddy_current/data_daily_processed/train/X",
                         y_dir= "/root/shared-nvme/AI_eddy_current/data_daily_processed/train/Y",
                         start_date="19930101", end_date="20171201")
    
    # Use 2 years as the validation set
    db_val = MyDataset(x_dir= "/root/shared-nvme/AI_eddy_current/data_daily_processed/train/X",
                       y_dir= "/root/shared-nvme/AI_eddy_current/data_daily_processed/train/Y",
                       start_date="20180101", end_date="20191201")
    
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of valid set is: {}".format(len(db_val)))
    # Set random seed
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)
    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    
    model.train()
    mse_loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999),eps=1e-08, weight_decay=0.0001) 

    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(train_loader)
    iterator = tqdm(range(max_epoch), ncols=70)
    best_loss = float('inf')
    eval_interval = 1

    train_losses = []
    val_losses = []

    # Training loop
    for epoch_num in iterator:
        model.train()
        batch_mse_loss = 0        

        for i_batch, (image_batch, label_batch) in tqdm(enumerate(train_loader), 
                                                    desc=f"Train: {epoch_num}", 
                                                    total=len(train_loader), 
                                                    leave=False):
            
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)
            loss = mse_loss(outputs, label_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Learning rate scheduling
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num += 1

            if iter_num % 20 == 0:
                print(f'Iteration: {iter_num}/{max_iterations}, LR: {lr_:.6f}, Loss: {loss.item():.4f}')

            batch_mse_loss += loss.item()

        batch_mse_loss /= len(train_loader)
        print(f'Epoch {epoch_num} - Train MSE Loss: {batch_mse_loss:.4f}')
        train_losses.append(batch_mse_loss)

        # Validation
        if (epoch_num + 1) % eval_interval == 0:
            model.eval()
            val_mse_loss = 0
            with torch.no_grad():
                for i_batch, (image_batch, label_batch) in tqdm(enumerate(val_loader), 
                                                            desc=f"Val: {epoch_num}", 
                                                            total=len(val_loader), 
                                                            leave=False):
                    image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                    outputs = model(image_batch)
                    loss = mse_loss(outputs, label_batch)
                    val_mse_loss += loss.item()

                val_mse_loss /= len(val_loader)
                print(f'Epoch {epoch_num} - Val MSE Loss: {val_mse_loss:.4f}')
                val_losses.append(val_mse_loss)

                if val_mse_loss < best_loss:
                    save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    best_loss = val_mse_loss
                    print(f"New best model saved at {save_mode_path} with loss {best_loss:.4f}")
                else:
                    save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    print(f"Model saved at {save_mode_path}")
    
        # Save loss records
        np.save(os.path.join(snapshot_path, 'train_losses.npy'), np.array(train_losses))
        np.save(os.path.join(snapshot_path, 'val_losses.npy'), np.array(val_losses))
        print(f"Training and validation losses saved to {snapshot_path}")

        plot_loss_curves(train_losses, val_losses, save_path=os.path.join(snapshot_path,'loss_curve.png'))

    return "Training Finished!"