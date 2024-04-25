Here is an example of the classes you requested:

```
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

class MetricsCalculator:
    def __init__(self):
        pass

    def pixel_accuracy(self, output, target):
        output = torch.argmax(output, dim=1)
        correct_pixels = (output == target).sum().item()
        total_pixels = target.shape[0] * target.shape[1] * target.shape[2]
        return correct_pixels / total_pixels

    def intersection_over_union(self, output, target):
        output = torch.argmax(output, dim=1)
        intersection = (output & target).sum().item()
        union = (output | target).sum().item()
        return intersection / union

    def dice_coefficient(self, output, target):
        output = torch.argmax(output, dim=1)
        intersection = (output & target).sum().item()
        total = (output.sum().item() + target.sum().item())
        return (2 * intersection) / total

class SegmentationDataset(Dataset):
    def __init__(self, masks_dir, results_dir):
        self.masks_dir = masks_dir
        self.results_dir = results_dir
        self.masks_files = os.listdir(masks_dir)
        self.results_files = os.listdir(results_dir)

    def __len__(self):
        return len(self.masks_files)

    def __getitem__(self, idx):
        mask = torch.load(os.path.join(self.masks_dir, self.masks_files[idx]))
        result = torch.load(os.path.join(self.results_dir, self.results_files[idx]))
        return mask, result

class MetricsPlotter:
    def __init__(self):
        pass

    def plot_metrics(self, metrics):
        plt.plot(metrics['pixel_accuracy'], label='Pixel Accuracy')
        plt.plot(metrics['intersection_over_union'], label='Intersection Over Union')
        plt.plot(metrics['dice_coefficient'], label='Dice Coefficient')
        plt.legend()
        plt.show()

# Example usage:

# Create a MetricsCalculator instance
calculator = MetricsCalculator()

# Create a SegmentationDataset instance
dataset = SegmentationDataset('path/to/masks', 'path/to/results')

# Create a DataLoader instance
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize metrics lists
pixel_accuracy_list = []
intersection_over_union_list = []
dice_coefficient_list = []

# Calculate metrics for each batch
for batch in dataloader:
    masks, results = batch
    pixel_accuracy = calculator.pixel_accuracy(results, masks)
    intersection_over_union = calculator.intersection_over_union(results, masks)
    dice_coefficient = calculator.dice_coefficient(results, masks)
    pixel_accuracy_list.append(pixel_accuracy)
    intersection_over_union_list.append(intersection_over_union)
    dice_coefficient_list.append(dice_coefficient)

# Create a MetricsPlotter instance
plotter = MetricsPlotter()

# Plot the metrics
metrics = {
    'pixel_accuracy': pixel_accuracy_list,
    'intersection_over_union': intersection_over_union_list,
    'dice_coefficient': dice_coefficient_list
}
plotter.plot_metrics(metrics)
```

This code defines three classes:

- `MetricsCalculator`: calculates the metrics (pixel accuracy, intersection over union, dice coefficient) between the model results and the ground truth masks.
- `SegmentationDataset`: loads the ground truth masks and the model results from directories.
- `MetricsPlotter`: plots the metrics.

The example usage at the end shows how to create instances of these classes, calculate the metrics for each batch in the DataLoader, and plot the metrics.
