import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np

# Dummy dataset and model for demonstration
class DummyDataset(torch.utils.data.Dataset):
    def __getitem__(self, index):
        # Generate random image and mask for demonstration purposes
        image = torch.randn(3, 256, 256)
        mask = torch.randint(0, 2, (256, 256))
        return image, mask

    def __len__(self):
        return 1000

class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 2, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Utility function to calculate IoU
def calculate_iou(pred, target, num_classes=2):
    ious = []
    pred = torch.argmax(pred, dim=1)
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).float().sum().item()
        union = (pred_inds | target_inds).float().sum().item()
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)

# Training function
def train(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(0)
    model = SimpleSegmentationModel().to(rank)
    model = DDP(model, device_ids=[rank])

    dataset = DummyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(10):
        sampler.set_epoch(epoch)
        epoch_loss = 0
        epoch_iou = 0
        for images, masks in dataloader:
            images = images.to(rank)
            masks = masks.to(rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_iou += calculate_iou(outputs, masks)

        print(f"Rank {rank}, Epoch [{epoch+1}/10], Loss: {epoch_loss/len(dataloader)}, IoU: {epoch_iou/len(dataloader)}")

    dist.destroy_process_group()

def main():
    world_size = 2
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True
