**Replication Code for: Reconstructing eddy-resolved subsurface currents using artificial intelligence**

We develop a Deep Learning-based Subsurface Current Reconstruction (DLSCR) model for the Indo-Pacific Convergence Zone (IPCZ; 99–150°E, 12°S–35°N). 
DLSCR ingests seven surface variables, including zonal and meridional surface currents (U, V), sea surface height (SSH), sea surface temperature (SST), sea surface salinity (SSS), and zonal and meridional wind stress (\tau_x, \tau_y), and then reconstruct subsurface ocean currents down to 643 m.

code/
└── src/
    ├── config.py
    ├── datasets.py
    ├── model.py
    ├── swin_transformer_unet_skip_expand_decoder_sys.py
    ├── train_ddp.py
    ├── trainer.py
    ├── test.py
    └── utils.py

**config.py**  
Configuration management: defines/loads hyperparameters and model/training settings (e.g., Swin params, batch size, image size, paths).

**datasets.py**
Dataset & dataloading utilities: implements Dataset classes to read daily .npy inputs/labels, reshape tensors, and handle optional preprocessing.

**model.py**
Model wrapper/entry point: exposes the SwinUnet module and connects configuration to the network for training/inference.

**swin_transformer_unet_skip_expand_decoder_sys.py**
Core Swin-Unet architecture: Swin Transformer encoder + U-Net-style skip connections + decoder (the main network implementation).

**train_ddp.py**
Distributed training script (DDP): initializes torch.distributed, uses DistributedSampler, runs train/val loops, saves checkpoints, supports multi-GPU via torchrun.

**trainer.py**
Non-DDP training pipeline: single-GPU or DataParallel training loop, optimizer/scheduler, logging, checkpoint saving, loss-curve tracking.

**test.py**
Evaluation/inference script: loads a trained checkpoint, runs forward passes on the test set, computes metrics (e.g., MSE), and saves predictions.

**utils.py**
Helper functions: denormalization, plotting (e.g., loss curves), and other shared utilities.
