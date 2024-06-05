import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        # Knowledge distillation loss
        kd_loss = self.criterion(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1)
        ) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        ce_loss = self.ce_loss(student_logits, targets)
        
        # Combined loss
        loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss
        return loss

class KnowledgeDistillationModel(nn.Module):
    def __init__(self, student_model, teacher_model, temperature=1.0, alpha=0.5):
        super(KnowledgeDistillationModel, self).__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.criterion = KnowledgeDistillationLoss(temperature, alpha)

    def forward(self, x, targets):
        student_logits = self.student_model(x)
        with torch.no_grad():
            teacher_logits = self.teacher_model(x)
        loss = self.criterion(student_logits, teacher_logits, targets)
        return student_logits, loss

def train(rank, world_size, args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Initialize models
    student_model = StudentModel().to(rank)
    teacher_model = TeacherModel().to(rank)
    
    # Wrap models with DDP
    student_model = DDP(student_model, device_ids=[rank])
    teacher_model = DDP(teacher_model, device_ids=[rank])
    
    # Load datasets and create DataLoader
    train_dataset = MyDataset()
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    
    # Create KnowledgeDistillationModel
    kd_model = KnowledgeDistillationModel(student_model, teacher_model).to(rank)
    
    # Optimizer
    optimizer = optim.Adam(kd_model.student_model.parameters(), lr=args.lr)
    
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(rank), targets.to(rank)
            
            optimizer.zero_grad()
            outputs, loss = kd_model(inputs, targets)
            loss.backward()
            optimizer.step()
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {loss.item()}")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    args = {
        "batch_size": 32,
        "lr": 0.001,
        "epochs": 10,
        "world_size": torch.cuda.device_count()
    }
    
    mp.spawn(train, args=(args["world_size"], args), nprocs=args["world_size"], join=True)
