import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SwinUnet as ViT_seg
from config import get_config
from datasets import MyDataset
from utils import Denormalizer


def test_model(args):
    config = get_config(args)

    # Set random seeds and cuDNN parameters
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Build the model
    model = ViT_seg(config, img_size=args.img_size, num_classes=args.num_classes).cuda()

    # New way to load weights (note that training saved model.module.state_dict())
    state_dict = torch.load(args.model_path, map_location="cuda")
    model.load_state_dict(state_dict, strict=True)  # Strict matching
    model.eval()

    # Build the test dataset
    db_test = MyDataset(
        x_dir="/root/shared-nvme/AI_eddy_current/data_daily_processed/test/X",
        y_dir="/root/shared-nvme/AI_eddy_current/data_daily_processed/test/Y",
        start_date="20200101",
        end_date="20221231"
    )

    test_loader = DataLoader(db_test,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             pin_memory=True)

    mse_loss = nn.MSELoss()
    total_mse = 0.0
    denormalizer = Denormalizer()

    with torch.no_grad():
        for idx, (image_batch, label_batch) in enumerate(tqdm(test_loader, desc="Testing", ncols=80)):
            image_batch = image_batch.cuda(non_blocking=True)
            label_batch = label_batch.cuda(non_blocking=True)

            outputs = model(image_batch)
            loss = mse_loss(outputs, label_batch)
            total_mse += loss.item()

            # If you need to save the predictions, you can enable it here
            batch_output = outputs.detach().cpu().numpy()
            batch_output = batch_output.reshape(batch_output.shape[0], 2, 22, 512, 512)
            batch_dates = db_test.dates[idx * test_loader.batch_size: (idx + 1) * test_loader.batch_size]
            for i, date_str in enumerate(batch_dates):
                output_path = os.path.join(args.output_dir, f"Y_u_v_{date_str}.npy")
                np.save(output_path, denormalizer.denormalize(batch_output[i]))

    avg_mse = total_mse / len(test_loader)
    print(f"Test MSE Loss: {avg_mse:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=44, help='output channel of network')
    parser.add_argument('--output_dir', type=str,
                        default='/root/shared-nvme/AI_eddy_current/SwinUnet_new/outputs', help='output dir')
    parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
    parser.add_argument('--img_size', type=int, default=512, help='input patch size of network input')
    parser.add_argument('--cfg', type=str,
                        default='/root/shared-nvme/AI_eddy_current/SwinUnet/src/swin_tiny_patch4_window7_512_lite.yaml',
                        required=False, metavar="FILE", help='path to config file')
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument('--model_path', type=str,
                        default='/root/shared-nvme/AI_eddy_current/SwinUnet_new/data/model_weights/best_model.pth',
                        help='Path to the trained model checkpoint')
    
    # parser.add_argument('--model_path', type=str,
    #                 default='/root/shared-nvme/AI_eddy_current/SwinUnet_new/data/model_weights_old/best_model.pth',
    #                 help='Path to the trained model checkpoint')
    # parser.add_argument('--n_gpu', type=int, default=2, help='total gpu')

    args = parser.parse_args()
    test_model(args)