import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

class DistributedTrainer:
    def __init__(self, model, dataset, batch_size, lr, num_epochs, world_size, rank):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.world_size = world_size
        self.rank = rank

        # Initialize the process group for DDP
        dist.init_process_group(backend='nccl', world_size=world_size, rank=rank)
        
        # Setup the model for distributed training
        self.model = self.model.to(rank)
        self.model = DDP(self.model, device_ids=[rank])
        
        # Setup the dataloader with DistributedSampler
        self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, num_replicas=world_size, rank=rank)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, sampler=self.sampler)
        
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().to(rank)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def train(self):
        for epoch in range(self.num_epochs):
            self.sampler.set_epoch(epoch)  # Shuffle data for each epoch
            for inputs, labels in self.dataloader:
                inputs = inputs.to(self.rank)
                labels = labels.to(self.rank)

                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

            if self.rank == 0:  # Only print progress from the main process
                print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item()}')

    def cleanup(self):
        dist.destroy_process_group()

# Example usage
if __name__ == "__main__":
    import torch.multiprocessing as mp
    from torchvision import datasets, transforms

    # Define model
    class SimpleModel(nn.Module):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.fc = nn.Linear(28*28, 10)

        def forward(self, x):
            x = x.view(-1, 28*28)
            return self.fc(x)

    # Define dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Set parameters
    batch_size = 64
    lr = 0.01
    num_epochs = 5
    world_size = 2

    def spawn_process(rank, world_size):
        model = SimpleModel()
        trainer = DistributedTrainer(model, train_dataset, batch_size, lr, num_epochs, world_size, rank)
        trainer.train()
        trainer.cleanup()

    mp.spawn(spawn_process, args=(world_size,), nprocs=world_size, join=True)
