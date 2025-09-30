import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from cnn_model import create_model

class ModelTrainer:
    """Utility class for training and evaluating CNN models."""
    
    def __init__(self, model, device, num_classes=7):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train the model for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """Validate the model for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001, 
              weight_decay=1e-4, patience=10):
        """Train the model with early stopping."""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validation
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if epoch % 5 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}/{epochs}:')
                print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
                print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
                print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
                print('-' * 50)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.train_losses, self.val_losses, self.train_accuracies, self.val_accuracies

class ModelEvaluator:
    """Utility class for evaluating trained models."""
    
    def __init__(self, model, device, class_names=None):
        self.model = model
        self.device = device
        self.class_names = class_names or [f'Class_{i}' for i in range(7)]
        
    def evaluate(self, test_loader):
        """Evaluate the model on test data."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def generate_classification_report(self, predictions, targets):
        """Generate detailed classification report."""
        return classification_report(targets, predictions, target_names=self.class_names)
    
    def plot_confusion_matrix(self, predictions, targets, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs, save_path=None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Training Loss', color='blue')
        ax1.plot(val_losses, label='Validation Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(train_accs, label='Training Accuracy', color='blue')
        ax2.plot(val_accs, label='Validation Accuracy', color='red')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def load_model(model_path, model_type='standard', num_classes=7, device='cpu'):
    """Load a trained model from checkpoint."""
    model = create_model(model_type, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def save_model(model, save_path, additional_info=None):
    """Save model with additional information."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_type': 'WorkerActivityCNN',
        'num_classes': 7,
        'class_names': ['sitting', 'standing', 'walking', 'working', 'resting', 'moving_objects', 'using_tools']
    }
    
    if additional_info:
        save_dict.update(additional_info)
    
    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    # Test the utilities
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = create_model('standard', num_classes=7)
    model = model.to(device)
    
    trainer = ModelTrainer(model, device)
    evaluator = ModelEvaluator(model, device)
    
    print("Model utilities loaded successfully!")
    print(f"Device: {device}")
    print(f"Model type: {type(model).__name__}")
