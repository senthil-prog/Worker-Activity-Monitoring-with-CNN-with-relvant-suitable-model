import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class WorkerActivityCNN(nn.Module):
    """
    CNN model for worker activity classification.
    
    Architecture:
    - Input: 3x224x224 RGB images
    - Multiple convolutional blocks with batch normalization and dropout
    - Global average pooling
    - Dense layers for classification
    - Output: 7 activity classes
    """
    
    def __init__(self, num_classes=7, input_channels=3):
        super(WorkerActivityCNN, self).__init__()
        
        # Convolutional blocks
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25)
        )
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """Extract feature maps for visualization."""
        features = []
        x = self.conv_block1(x)
        features.append(x)
        x = self.conv_block2(x)
        features.append(x)
        x = self.conv_block3(x)
        features.append(x)
        x = self.conv_block4(x)
        features.append(x)
        return features

class LightweightCNN(nn.Module):
    """
    Lightweight CNN for faster inference on resource-constrained devices.
    """
    
    def __init__(self, num_classes=7, input_channels=3):
        super(LightweightCNN, self).__init__()
        
        # Simplified architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def create_model(model_type='standard', num_classes=7, pretrained=False):
    """
    Factory function to create different model types.
    
    Args:
        model_type (str): Type of model ('standard' or 'lightweight')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    
    Returns:
        torch.nn.Module: The created model
    """
    if model_type == 'standard':
        model = WorkerActivityCNN(num_classes=num_classes)
    elif model_type == 'lightweight':
        model = LightweightCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if pretrained:
        # Load pretrained weights if available
        print(f"Loading pretrained weights for {model_type} model...")
        # This would load from a checkpoint file
    
    return model

def print_model_summary(model, input_size=(3, 224, 224)):
    """Print model architecture summary."""
    try:
        summary(model, input_size)
    except:
        print("Model Summary:")
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create and test standard model
    model = create_model('standard', num_classes=7)
    model = model.to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    output = model(dummy_input)
    
    print(f"Model output shape: {output.shape}")
    print(f"Model device: {next(model.parameters()).device}")
    
    # Print model summary
    print_model_summary(model)
