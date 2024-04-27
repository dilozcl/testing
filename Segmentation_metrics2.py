import numpy as np
import torch
from sklearn.metrics import confusion_matrix

def calculate_metrics(predictions, targets, num_classes):
    # Flatten predictions and targets
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Ignore background class (class 0)
    mask = targets != 0
    predictions = predictions[mask]
    targets = targets[mask]

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy(), labels=np.arange(1, num_classes))

    # Calculate true positives, false positives, and false negatives
    true_positives = np.diag(conf_matrix)
    false_positives = np.sum(conf_matrix, axis=1) - true_positives
    false_negatives = np.sum(conf_matrix, axis=0) - true_positives

    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-9)

    # Calculate mean precision, recall, and F1 score
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1_score = np.mean(f1_score)

    return {
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1_score': mean_f1_score
    }

# Example usage:
# Assuming you have predictions and targets as PyTorch tensors
predictions = torch.randint(0, 100, (100, 256, 256))  # Example predictions (batch_size, height, width)
targets = torch.randint(0, 100, (100, 256, 256))  # Example targets (batch_size, height, width)
num_classes = 100  # Number of segmentation classes (excluding background)

# Calculate metrics
metrics = calculate_metrics(predictions, targets, num_classes)
print(metrics)
