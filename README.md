# Replication Code for: **Reconstructing eddy-resolved subsurface currents using artificial intelligence**

We develop a Deep Learning-based Subsurface Current Reconstruction (**DLSCR**) model for the **Indo-Pacific Convergence Zone (IPCZ; 99–150°E, 12°S–35°N)**.  
DLSCR ingests **seven surface variables**—zonal and meridional surface currents (**U, V**), sea surface height (**SSH**), sea surface temperature (**SST**), sea surface salinity (**SSS**), and zonal/meridional wind stress (**τx, τy**)—and reconstructs **subsurface ocean currents down to 643 m**.

---

## Repository Structure

```text
code/
└── src/
    ├── config.py #  Configuration management, hyperparameters and model/training settings
    ├── datasets.py # Dataset & dataloader utilities
    ├── model.py # the Swin-Unet model setup
    ├── swin_transformer_unet_skip_expand_decoder_sys.py # Swin Transformer encoder + U-Net-style skip connections + decoder (main architecture).
    ├── train_ddp.py # Distributed training script
    ├── trainer.py #  Non-DDP training pipeline
    ├── test.py # inference script
    └── utils.py #  denormalization and other helper functions
