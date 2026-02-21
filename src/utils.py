import os
import glob
import re
import numpy as np
from scipy.ndimage import zoom

class Denormalizer:
    def __init__(self):
        self.mean_array, self.std_array = self.read_std_mean_Y()

    @staticmethod
    def extract_depth(filepath):
        """Extract depth from the file path (match float numbers)."""
        match = re.search(r'depth_([\d.]+)m', filepath)
        return float(match.group(1)) if match else float('inf')

    def read_std_mean_Y(self):
        """Read the mean and standard deviation for Y."""
        base_dirs = {
            'u': '/gpu-data/AI_eddy_current/statistic_coor/subsurface/u',
            'v': '/gpu-data/AI_eddy_current/statistic_coor/subsurface/v'
        }

        mean_list = []
        std_list = []

        for var in ['u', 'v']:
            file_list = sorted([
                f for f in glob.glob(os.path.join(base_dirs[var], '*.npy'))
                if 'depth_0.49m.npy' not in os.path.basename(f)
            ])

            file_list_sorted = sorted(file_list, key=self.extract_depth)

            for npyfile in file_list_sorted:
                data = np.load(npyfile, allow_pickle=True).item()
                mean_list.append(data['mean'])  # shape: (565, 613)
                std_list.append(data['std'])    # shape: (565, 613)

        mean_array = np.stack(mean_list, axis=0).reshape(2, 22, 565, 613)
        std_array = np.stack(std_list, axis=0).reshape(2, 22, 565, 613)

        return mean_array, std_array

    @staticmethod
    def resize_with_zoom(data, target_shape):
        """Resize data to the target shape."""
        assert data.shape[0] == target_shape[0], "Number of channels must match"
        zoom_factors = (1, 1, target_shape[2] / data.shape[2], target_shape[3] / data.shape[3])
        return zoom(data, zoom_factors, order=1)

    def denormalize(self, data_norm):
        """Denormalization operation."""
        data_resized = self.resize_with_zoom(data_norm, self.mean_array.shape)
        return data_resized * self.std_array + self.mean_array
    

import numpy as np
import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, val_losses, eval_interval=1, save_path=None):
    """
    Visualize training and validation loss curves.
    """
    # Generate epoch arrays
    epochs = np.arange(1, len(train_losses) + 1)
    val_epochs = np.arange(1, len(val_losses) + 1) * eval_interval  # Adjust based on eval_interval

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', color='blue', linewidth=2)
    plt.plot(val_epochs, val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Loss plot saved to {save_path}")
    else:
        plt.show()