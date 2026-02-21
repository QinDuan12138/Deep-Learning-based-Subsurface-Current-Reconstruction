# Read npy data X (7 × 512 × 512) and Y (44 × 512 × 512)
import os
import glob
import numpy as np
from scipy.ndimage import zoom
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime, timedelta

class MyDataset(Dataset):
    def __init__(self, x_dir, y_dir, start_date="19930101", end_date="20191231"):
        self.x_dir = x_dir
        self.y_dir = y_dir

        # Generate all dates from start_date to end_date
        self.dates = []
        current = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            x_path = os.path.join(x_dir, f"X_u_v_ssh_sst_sss_taux_tauy_{date_str}.npy")
            y_path = os.path.join(y_dir, f"Y_u_v_{date_str}.npy")
            if os.path.exists(x_path) and os.path.exists(y_path):
                self.dates.append(date_str)
            current += timedelta(days=1)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date_str = self.dates[idx]

        # Load data
        x_path = os.path.join(self.x_dir, f"X_u_v_ssh_sst_sss_taux_tauy_{date_str}.npy")
        y_path = os.path.join(self.y_dir, f"Y_u_v_{date_str}.npy")

        X = np.load(x_path)  # shape: (7, 512, 512)
        Y = np.load(y_path)  # shape: (2, 22, 512, 512)
        Y = Y.reshape(44, 512, 512)  # Merge the first two dimensions

        # Convert to Tensor
        X_final = torch.from_numpy(X).float()
        Y_final = torch.from_numpy(Y).float()

        return X_final, Y_final
    

# Mapping from variable name to channel index
VAR_INDEX_MAP = {
    'u': 0,
    'v': 1,
    'ssh': 2,
    'sst': 3,
    'sss': 4,
    'taux': 5,
    'tauy': 6
}

class MyDataset_experiment(Dataset):
    def __init__(self, x_dir, y_dir, var_name=None, start_date="19930101", end_date="20191231"):
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.var_name = var_name  # e.g., "u", "v", ..., or None if no replacement

        # Validate variable name
        if self.var_name is not None and self.var_name not in VAR_INDEX_MAP:
            raise ValueError(f"Invalid variable name '{self.var_name}', must be one of {list(VAR_INDEX_MAP.keys())}")

        # Load climatology data
        self.clim_data = np.load('/root/shared-nvme/AI_eddy_current/statistic_coor/processed_climatology_u_v_ssh_sst_sss_taux_tauy.npy')  # shape: (12, 7, 512, 512)

        # Generate date list
        self.dates = []
        current = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            x_path = os.path.join(x_dir, f"X_u_v_ssh_sst_sss_taux_tauy_{date_str}.npy")
            y_path = os.path.join(y_dir, f"Y_u_v_{date_str}.npy")
            if os.path.exists(x_path) and os.path.exists(y_path):
                self.dates.append(date_str)
            current += timedelta(days=1)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date_str = self.dates[idx]

        # Load original data
        x_path = os.path.join(self.x_dir, f"X_u_v_ssh_sst_sss_taux_tauy_{date_str}.npy")
        y_path = os.path.join(self.y_dir, f"Y_u_v_{date_str}.npy")

        X = np.load(x_path)  # shape: (7, 512, 512)
        Y = np.load(y_path)  # shape: (2, 22, 512, 512)
        Y = Y.reshape(44, 512, 512)  # shape: (44, 512, 512)

        # Replace specified channel with climatology values
        if self.var_name is not None:
            month_idx = int(date_str[4:6]) - 1  # 0-based month index
            var_idx = VAR_INDEX_MAP[self.var_name]
            X[var_idx] = self.clim_data[month_idx, var_idx]

        # Convert to Tensor
        X_final = torch.from_numpy(X).float()
        Y_final = torch.from_numpy(Y).float()

        return X_final, Y_final
    
    
class MyDataset_sattlite(Dataset):
    def __init__(self, x_dir, y_dir, start_date="19930101", end_date="20191231"):
        self.x_dir = x_dir
        self.y_dir = y_dir

        # Generate all dates from start_date to end_date
        self.dates = []
        current = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            x_path = os.path.join(x_dir, f"X_u_v_ssh_sst_sss_taux_tauy_{date_str}.npy")
            y_path = os.path.join(y_dir, f"Y_u_v_{date_str}.npy")
            if os.path.exists(x_path) and os.path.exists(y_path):
                self.dates.append(date_str)
            current += timedelta(days=1)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date_str = self.dates[idx]

        # Load data
        x_path = os.path.join(self.x_dir, f"X_u_v_ssh_sst_sss_taux_tauy_{date_str}.npy")
        y_path = os.path.join(self.y_dir, f"Y_u_v_{date_str}.npy")

        X = np.load(x_path)  # original shape: (7, 565, 613)
        Y = np.load(y_path)  # (2, 22,  512, 512)
        Y = Y.reshape(44, Y.shape[2], Y.shape[3])  # (44, 512, 512)

        # Convert to Tensor
        X_tensor = torch.from_numpy(X).unsqueeze(0).float()  # (1, 7, 565, 613)
        Y_tensor = torch.from_numpy(Y).float()  # (44, 512, 512)

        # Interpolate/resize X -> (7, 512, 512)
        X_resized = F.interpolate(X_tensor, size=(512, 512), mode="bilinear", align_corners=False)
        X_final = X_resized.squeeze(0)  # (7, 512, 512)

        return X_final, Y_tensor
    
class MyDataset_experiment_filter(Dataset):
    def __init__(self, x_dir, xf_dir, y_dir, var_name=None,
                 start_date="19930101", end_date="20191231"):

        self.x_dir = x_dir
        self.xf_dir = xf_dir
        self.y_dir = y_dir
        self.var_name = var_name

        if self.var_name is not None and self.var_name not in VAR_INDEX_MAP:
            raise ValueError(f"Invalid variable name '{self.var_name}', must be one of {list(VAR_INDEX_MAP.keys())}")

        # Determine highpass / lowpass (normalize path to avoid trailing '/' issues)
        xf_dir_norm = os.path.normpath(xf_dir)
        self.xf_mode = os.path.basename(xf_dir_norm)  # "highpass" or "lowpass"
        if self.xf_mode not in ("highpass", "lowpass"):
            raise ValueError(f"xf_dir should end with 'highpass' or 'lowpass', got: {xf_dir}")

        # Save prefix for __getitem__
        self.xf_prefix = f"X_u_v_ssh_sst_sss_taux_tauy_{self.xf_mode}_"

        self.dates = []
        current = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")

        while current <= end:
            date_str = current.strftime("%Y%m%d")
            x_path  = os.path.join(x_dir,  f"X_u_v_ssh_sst_sss_taux_tauy_{date_str}.npy")
            xf_path = os.path.join(xf_dir, f"{self.xf_prefix}{date_str}.npy")
            y_path  = os.path.join(y_dir,  f"Y_u_v_{date_str}.npy")

            if os.path.exists(x_path) and os.path.exists(xf_path) and os.path.exists(y_path):
                self.dates.append(date_str)

            current += timedelta(days=1)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date_str = self.dates[idx]

        # Load original data
        x_path  = os.path.join(self.x_dir, f"X_u_v_ssh_sst_sss_taux_tauy_{date_str}.npy")
        xf_path = os.path.join(self.xf_dir, f"{self.xf_prefix}{date_str}.npy")
        y_path  = os.path.join(self.y_dir, f"Y_u_v_{date_str}.npy")

        X  = np.load(x_path)    # (7, 512, 512) original
        Xf = np.load(xf_path)   # (7, 512, 512) filtered (lowpass or highpass)
        Y  = np.load(y_path)    # (2, 22, 512, 512)
        Y  = Y.reshape(44, 512, 512)

        # Replace specified channel with the corresponding filtered channel
        if self.var_name is not None:
            var_idx = VAR_INDEX_MAP[self.var_name]
            X[var_idx] = Xf[var_idx]

        # Convert to Tensor
        X_final = torch.from_numpy(X).float()
        Y_final = torch.from_numpy(Y).float()

        return X_final, Y_final

class MyDataset_experiment_lhpass_only(Dataset):
    def __init__(self, xf_dir, y_dir, var_name=None,
                 start_date="19930101", end_date="20191231"):
        self.xf_dir = xf_dir
        self.y_dir = y_dir
        
        # Determine highpass / lowpass (normalize path to avoid trailing '/' issues)
        xf_dir_norm = os.path.normpath(xf_dir)
        self.xf_mode = os.path.basename(xf_dir_norm)  # "highpass" or "lowpass"
        if self.xf_mode not in ("highpass", "lowpass"):
            raise ValueError(f"xf_dir should end with 'highpass' or 'lowpass', got: {xf_dir}")

        # Save prefix for __getitem__
        self.xf_prefix = f"X_u_v_ssh_sst_sss_taux_tauy_{self.xf_mode}_"

        # Generate all dates from start_date to end_date
        self.dates = []
        current = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        while current <= end:
            date_str = current.strftime("%Y%m%d")
            xf_path = os.path.join(xf_dir, f"{self.xf_prefix}{date_str}.npy")
            y_path = os.path.join(y_dir, f"Y_u_v_{date_str}.npy")
            if os.path.exists(xf_path) and os.path.exists(y_path):
                self.dates.append(date_str)
            current += timedelta(days=1)

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date_str = self.dates[idx]

        # Load data
        xf_path = os.path.join(self.xf_dir, f"{self.xf_prefix}{date_str}.npy")
        y_path = os.path.join(self.y_dir, f"Y_u_v_{date_str}.npy")

        X = np.load(xf_path)  # shape: (7, 512, 512)
        Y = np.load(y_path)  # shape: (2, 22, 512, 512)
        Y = Y.reshape(44, 512, 512)  # Merge the first two dimensions

        # Convert to Tensor
        X_final = torch.from_numpy(X).float()
        Y_final = torch.from_numpy(Y).float()

        return X_final, Y_final        

if __name__ == "__main__":
    
    import time
    total_start = time.time()  # Record total start time

    # Training data paths
    x_dir = "/root/shared-nvme/AI_eddy_current/data_daily_processed/test/X"
    y_dir = "/root/shared-nvme/AI_eddy_current/data_daily_processed/test/Y"

    # Initialize dataset
    dataset = MyDataset(x_dir, y_dir, start_date="20200101", end_date="20221231")
    print("Total samples:", len(dataset))

    # X_final, Y_final = dataset[0]
    # print(X_final.shape)
    # print(Y_final.shape)

    DataLoader_start = time.time()
    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)

    DataLoader_elapsed = time.time() - DataLoader_start
    print(f"\n[DataLoader elapsed Time] {DataLoader_elapsed:.2f}s")

    batch_start = time.time()
    # Fetch one batch and print shapes
    batch_num = 0
    for batch in dataloader:
        X_batch, Y_batch = batch  # X: (B, 7, 565, 613), Y: (B, 44, 565, 613)
        print("X batch shape:", X_batch.shape)
        print("Y batch shape:", Y_batch.shape)
        batch_num += 1
    print(f'batch num = {batch_num-1}')    
    
    batch_elapsed = time.time() - batch_start
    print(f"\n[batch elapsed Time] {batch_elapsed:.2f}s")

    # ===== Total elapsed time =====
    total_elapsed = time.time() - total_start
    print(f"\n[Total Execution Time] {total_elapsed:.2f}s")