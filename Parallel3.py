import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torchvision import transforms
from sklearn.metrics import jaccard_score
import numpy as np
from tqdm import tqdm

# Example dataset and model
class ExampleDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        # Initialization code (e.g., loading data)
        self.transform = transform

    def __len__(self):
        # Length of the dataset
        return 100

    def __getitem__(self, idx):
        # Return a sample and its corresponding label
        image = np.random.rand(3, 256, 256).astype(np.float32)
        mask = np.random.randint(0, 2, (256, 256)).astype(np.int64)
        if self.transform:
            image = self.transform(image)
        return image, mask

class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes):
        super(SimpleSegmentationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# IOU Calculation
def calculate_iou(pred, target, num_classes):
    iou_list = []
    pred = pred.argmax(dim=1)
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    for cls in range(num_classes):
        pred_mask = (pred_np == cls)
        target_mask = (target_np == cls)
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        if union == 0:
            iou = float('nan')  # If there is no ground truth, do not include in the IoU
        else:
            iou = intersection / union
        iou_list.append(iou)
    return np.nanmean(iou_list)

# Training loop
def train(rank, world_size):
    # Initialize the process group
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    # Create the model and move it to GPU with id rank
    model = SimpleSegmentationModel(num_classes=2).to(rank)
    model = DDP(model, device_ids=[rank])

    # Create dataset and dataloader
    transform = transforms.ToTensor()
    dataset = ExampleDataset(transform=transform)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler, num_workers=4, pin_memory=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        model.train()
        sampler.set_epoch(epoch)
        epoch_loss = 0.0
        iou_score = 0.0

        for images, masks in tqdm(dataloader, desc=f"Rank {rank} Epoch {epoch+1}"):
            images, masks = images.to(rank), masks.to(rank)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            iou_score += calculate_iou(outputs, masks, num_classes=2)

        print(f"Rank {rank} Epoch [{epoch+1}/10], Loss: {epoch_loss/len(dataloader)}, IoU: {iou_score/len(dataloader)}")

    # Cleanup
    dist.destroy_process_group()

# Main entry point
def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
