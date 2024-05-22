import torch

def calculate_iou(predictions, targets, num_classes):
    """
    Calculate the Intersection over Union (IoU) for multi-class segmentation.
    
    Args:
        predictions (torch.Tensor): Predicted one-hot encoded tensor of shape (N, C, H, W).
        targets (torch.Tensor): Ground truth one-hot encoded tensor of shape (N, C, H, W).
        num_classes (int): Number of classes (C).
    
    Returns:
        torch.Tensor: IoU for each class.
    """
    # Ensure the input tensors are on the same device
    device = predictions.device
    
    # Initialize intersection and union tensors
    intersection = torch.zeros(num_classes, device=device)
    union = torch.zeros(num_classes, device=device)
    
    # Calculate intersection and union for each class
    for cls in range(num_classes):
        pred_cls = predictions[:, cls, :, :]
        target_cls = targets[:, cls, :, :]
        
        # Intersection is the sum of element-wise multiplication
        intersection[cls] = (pred_cls * target_cls).sum()
        
        # Union is the sum of the predicted and target elements minus the intersection
        union[cls] = pred_cls.sum() + target_cls.sum() - intersection[cls]
    
    # Compute IoU for each class
    iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
    return iou

# Example usage:
# Assume we have a batch of size 2, with 3 classes, and image size 4x4
predictions = torch.randint(0, 2, (2, 3, 4, 4)).float().cuda()  # Example predictions (one-hot encoded)
targets = torch.randint(0, 2, (2, 3, 4, 4)).float().cuda()       # Example targets (one-hot encoded)

num_classes = 3  # Number of classes

# Calculate IoU
iou = calculate_iou(predictions, targets, num_classes)
print("IoU for each class:", iou)
