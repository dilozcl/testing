import torch
import torch.nn as nn
import torch.nn.functional as F

class SegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, decoder_channels=256):
        super(SegmentationHead, self).__init__()
        
        # Decoder layers
        self.conv1 = nn.Conv2d(in_channels, decoder_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(decoder_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(decoder_channels)
        
        self.conv3 = nn.Conv2d(decoder_channels, num_classes, kernel_size=1) # Output layer
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False) # Adjust scale_factor if needed
        return x

# Example usage
class SemanticSegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes):
        super(SemanticSegmentationModel, self).__init__()
        self.encoder = encoder
        self.decoder = SegmentationHead(in_channels=encoder.out_channels, num_classes=num_classes)
    
    def forward(self, x):
        features = self.encoder(x)
        seg_map = self.decoder(features)
        return seg_map

# Loss function for multi-class segmentation
def compute_loss(predictions, targets):
    return F.cross_entropy(predictions, targets)

# Assuming you have an encoder backbone ready
class DummyEncoder(nn.Module):
    def __init__(self):
        super(DummyEncoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_channels = 128
    
    def forward(self, x):
        return self.features(x)

# Dummy encoder instance
encoder = DummyEncoder()
num_classes = 21  # Example for 21 classes, adjust accordingly

# Instantiate the model
model = SemanticSegmentationModel(encoder, num_classes)

# Example input
input_tensor = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image
output = model(input_tensor)

print(output.shape)  # Should output torch.Size([1, num_classes, H, W])
