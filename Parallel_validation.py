import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

class SegmentationTrainer:
    def __init__(self, model, train_dataset, val_dataset, num_epochs, batch_size, lr, device, world_size, rank):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.world_size = world_size
        self.rank = rank

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.model = DDP(self.model, device_ids=[rank])

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)
        self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=world_size, rank=rank)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, sampler=self.train_sampler, num_workers=4, pin_memory=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, sampler=self.val_sampler, num_workers=4, pin_memory=True)

    def train_one_epoch(self, epoch):
        self.model.train()
        self.train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.train_loader.dataset)
        print(f'Train Epoch {epoch}, Loss: {epoch_loss:.4f}')

    def validate_one_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(self.val_loader.dataset)
        print(f'Validation Epoch {epoch}, Loss: {epoch_loss:.4f}')

    def train(self):
        for epoch in range(self.num_epochs):
            self.train_one_epoch(epoch)
            self.validate_one_epoch(epoch)

def init_process(rank, world_size, model, train_dataset, val_dataset, num_epochs, batch_size, lr, backend='nccl'):
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    torch.manual_seed(42)
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    trainer = SegmentationTrainer(model, train_dataset, val_dataset, num_epochs, batch_size, lr, device, world_size, rank)
    trainer.train()

    dist.destroy_process_group()

if __name__ == '__main__':
    import argparse
    from model import MySegmentationModel  # Import your model definition

    parser = argparse.ArgumentParser()
    parser.add_argument('--world_size', type=int, default=1, help='Number of distributed processes')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    world_size = args.world_size
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr

    train_dataset = datasets.Cityscapes(root='data', split='train', mode='fine', target_type='semantic', transform=transforms.ToTensor())
    val_dataset = datasets.Cityscapes(root='data', split='val', mode='fine', target_type='semantic', transform=transforms.ToTensor())

    model = MySegmentationModel()  # Replace with your model

    dist.init_process_group('nccl')
    processes = []

    for rank in range(world_size):
        p = torch.multiprocessing.Process(target=init_process, args=(rank, world_size, model, train_dataset, val_dataset, num_epochs, batch_size, lr))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
