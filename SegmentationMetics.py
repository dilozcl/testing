import os
import cv2
import numpy as np

class SegmentationMetrics:
    def __init__(self, mask_dir):
        self.mask_dir = mask_dir

    def calculate_iou(self, pred_mask, true_mask):
        intersection = np.logical_and(pred_mask, true_mask)
        union = np.logical_or(pred_mask, true_mask)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def calculate_dice_coefficient(self, pred_mask, true_mask):
        intersection = np.sum(np.logical_and(pred_mask, true_mask))
        dice_coefficient = (2. * intersection) / (np.sum(pred_mask) + np.sum(true_mask))
        return dice_coefficient

    def calculate_metrics(self, pred_mask_path):
        true_mask_name = os.path.basename(pred_mask_path).split('.')[0] + "_true.png"
        true_mask_path = os.path.join(self.mask_dir, true_mask_name)

        pred_mask = cv2.imread(pred_mask_path, cv2.IMREAD_GRAYSCALE)
        true_mask = cv2.imread(true_mask_path, cv2.IMREAD_GRAYSCALE)

        # Binarize masks
        pred_mask = (pred_mask > 0).astype(np.uint8)
        true_mask = (true_mask > 0).astype(np.uint8)

        iou = self.calculate_iou(pred_mask, true_mask)
        dice = self.calculate_dice_coefficient(pred_mask, true_mask)

        return iou, dice

# Example usage:
mask_dir = "/path/to/masks/directory"
metrics_calculator = SegmentationMetrics(mask_dir)

# Assuming you have predicted masks stored in a directory
pred_mask_paths = ["pred_mask_1.png", "pred_mask_2.png", ...]

iou_scores = []
dice_coefficients = []

for pred_mask_path in pred_mask_paths:
    iou, dice = metrics_calculator.calculate_metrics(pred_mask_path)
    iou_scores.append(iou)
    dice_coefficients.append(dice)

average_iou = np.mean(iou_scores)
average_dice = np.mean(dice_coefficients)

print("Average IoU:", average_iou)
print("Average Dice Coefficient:", average_dice)
