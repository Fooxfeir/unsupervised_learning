import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
import math

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create output directories
os.makedirs('./outputs', exist_ok=True)
os.makedirs('./outputs/crops', exist_ok=True)
os.makedirs('./outputs/umap', exist_ok=True)

class CustomImageDataset(Dataset):
    """Custom dataset for loading images with class labels from filenames"""
    def __init__(self, data_dir, transform=None, image_size=224):
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size
        
        # Find all PNG files
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        self.image_paths.sort()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG files found in {data_dir}")
        
        # Extract class labels from filenames (XXXX_YYYY.png format)
        self.labels = []
        for path in self.image_paths:
            filename = os.path.basename(path)
            try:
                class_num = int(filename.split('_')[0])
                self.labels.append(class_num - 1)  # Convert to 0-based indexing
            except (ValueError, IndexError):
                # If filename doesn't match expected format, use hash-based labeling
                class_num = hash(filename.split('.')[0]) % 10
                self.labels.append(class_num)
        
        unique_classes = len(set(self.labels))
        print(f"Found {len(self.image_paths)} images across {unique_classes} classes")
        print(f"Classes: {sorted(set(self.labels))}")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, idx

class ImprovedCNNEncoder(nn.Module):
    """Improved CNN Encoder with better architecture"""
    def __init__(self, input_channels=3):
        super().__init__()
        
        # More robust architecture inspired by ResNet
        self.features = nn.Sequential(
            # Initial conv block
            nn.Conv2d(input_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),  # 224 -> 56
            
            # Block 1
            self._make_layer(64, 64, stride=1),
            self._make_layer(64, 64, stride=1),
            
            # Block 2  
            self._make_layer(64, 128, stride=2),  # 56 -> 28
            self._make_layer(128, 128, stride=1),
            
            # Block 3
            self._make_layer(128, 256, stride=2),  # 28 -> 14
            self._make_layer(256, 256, stride=1),
            
            # Block 4
            self._make_layer(256, 512, stride=2),  # 14 -> 7
            self._make_layer(512, 512, stride=1),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        self.feature_dim = 512
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _make_layer(self, in_channels, out_channels, stride):
        """Create a residual-like block"""
        layers = []
        if stride != 1 or in_channels != out_channels:
            # Downsample if needed
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.features(x)

class DINOProjectionHead(nn.Module):
    """DINO projection head with proper normalization"""
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=65536, nlayers=3):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        
        layers.append(nn.Linear(hidden_dim, out_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Last layer normalization (important for DINO)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if hasattr(self.last_layer, 'weight_g'):
            self.last_layer.weight_g.requires_grad = False
        
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class FixedMultiCropDINO(nn.Module):
    """Fixed Multi-Crop DINO with correct implementation"""
    def __init__(self, input_channels=3, out_dim=65536):
        super().__init__()
        
        # Student and teacher networks
        self.student_encoder = ImprovedCNNEncoder(input_channels)
        self.teacher_encoder = ImprovedCNNEncoder(input_channels)
        
        feature_dim = self.student_encoder.feature_dim
        self.student_head = DINOProjectionHead(feature_dim, 2048, out_dim)
        self.teacher_head = DINOProjectionHead(feature_dim, 2048, out_dim)
        
        # Initialize teacher with student weights
        self._initialize_teacher()
        
        # Disable gradients for teacher
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_head.parameters():
            param.requires_grad = False
            
        # DINO centering mechanism - start with zeros
        self.register_buffer('center', torch.zeros(1, out_dim))
        
        print(f"Initialized Fixed DINO model:")
        print(f"  Encoder feature dim: {feature_dim}")
        print(f"  Projection output dim: {out_dim}")
        
    def _initialize_teacher(self):
        """Initialize teacher with student weights"""
        for student_param, teacher_param in zip(self.student_encoder.parameters(), 
                                              self.teacher_encoder.parameters()):
            teacher_param.data.copy_(student_param.data)
        
        for student_param, teacher_param in zip(self.student_head.parameters(), 
                                              self.teacher_head.parameters()):
            teacher_param.data.copy_(student_param.data)
    
    def update_teacher(self, momentum):
        """Update teacher with exponential moving average"""
        with torch.no_grad():
            for student_param, teacher_param in zip(self.student_encoder.parameters(), 
                                                  self.teacher_encoder.parameters()):
                teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
            
            for student_param, teacher_param in zip(self.student_head.parameters(), 
                                                  self.teacher_head.parameters()):
                teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1 - momentum)
    
    def update_center(self, teacher_output, momentum=0.9):
        """Update center for teacher centering with proper momentum"""
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(momentum).add_(batch_center, alpha=1 - momentum)
    
    def forward(self, crops):
        """Forward pass through both networks"""
        # Student processes all crops
        student_features = self.student_encoder(crops)
        student_output = self.student_head(student_features)
        
        # Teacher processes all crops (but no gradients)
        with torch.no_grad():
            teacher_features = self.teacher_encoder(crops)
            teacher_output = self.teacher_head(teacher_features)
        
        return student_output, teacher_output, student_features
    
    def encode_images(self, imgs):
        """Extract features for evaluation (use student encoder)"""
        with torch.no_grad():
            features = self.student_encoder(imgs)
        return features

class FixedMultiCropAugmentation:
    """Fixed multi-crop augmentation with proper parameters"""
    def __init__(self, global_crops_scale=(0.4, 1.0), local_crops_scale=(0.05, 0.4), 
                 local_crops_number=8, global_crops_size=224, local_crops_size=96):
        
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        
        # Stronger global augmentation
        self.global_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(global_crops_size, scale=global_crops_scale, 
                                       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Weaker local augmentation
        self.local_augmentation = transforms.Compose([
            transforms.RandomResizedCrop(local_crops_size, scale=local_crops_scale,
                                       interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __call__(self, image):
        """Generate multi-crop augmentations for a single image"""
        if torch.is_tensor(image):
            # Proper denormalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(image.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(image.device)
            # Correct denormalization formula
            image = torch.clamp(image * std + mean, 0, 1)
            image = transforms.ToPILImage()(image.cpu())
        
        # Generate all crops at their natural sizes first
        global_crops = [self.global_augmentation(image) for _ in range(2)]
        local_crops_original = [self.local_augmentation(image) for _ in range(self.local_crops_number)]
        
        # Resize local crops to global size for network processing
        resize_and_normalize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.global_crops_size, self.global_crops_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply resize to local crops
        local_crops_resized = []
        for local_crop in local_crops_original:
            # Denormalize first
            denorm_crop = local_crop * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            denorm_crop = torch.clamp(denorm_crop, 0, 1)
            
            # Resize and renormalize
            resized_crop = resize_and_normalize(denorm_crop)
            local_crops_resized.append(resized_crop)
        
        # Combine all crops
        all_crops = global_crops + local_crops_resized
        return torch.stack(all_crops)

def fixed_dino_loss(student_output, teacher_output, center, epoch=0, max_epochs=100,
                   student_temp=0.1, teacher_temp=0.04):
    """Fixed DINO loss with proper asymmetric structure"""
    
    batch_size = student_output.shape[0] // (2 + 8)  # 2 global + 8 local per image
    
    # Reshape outputs: [batch_size, n_crops, feature_dim]
    student_out = student_output.view(batch_size, -1, student_output.shape[1])
    teacher_out = teacher_output.view(batch_size, -1, teacher_output.shape[1])
    
    # Split global vs local crops
    teacher_global = teacher_out[:, :2]  # First 2 crops are global (teacher)
    student_local = student_out[:, 2:]   # Last 8 crops are local (student)
    
    # Apply centering schedule (gradually increase centering)
    centering_schedule = min(1.0, epoch / (max_epochs * 0.1))  # Reach full centering at 10% of training
    
    # Teacher outputs (global crops) with centering and temperature
    teacher_centered = (teacher_global - centering_schedule * center) / teacher_temp
    teacher_targets = F.softmax(teacher_centered, dim=-1)
    
    # Student outputs (local crops) with temperature
    student_out_temp = student_local / student_temp
    student_log_probs = F.log_softmax(student_out_temp, dim=-1)
    
    # Compute cross-entropy: each local crop predicts each global crop
    total_loss = 0
    n_pairs = 0
    
    for i in range(2):  # For each global crop (teacher)
        for j in range(8):  # For each local crop (student)  
            # Cross-entropy between student local crop j and teacher global crop i
            loss = -torch.sum(teacher_targets[:, i] * student_log_probs[:, j], dim=-1)
            total_loss += loss.mean()
            n_pairs += 1
    
    return total_loss / n_pairs

def create_multi_crop_batch(batch, multi_crop_fn):
    """Create multi-crop batch for DINO training"""
    all_crops = []
    
    for img in batch:
        crops = multi_crop_fn(img)  # Returns stacked tensor [10, 3, H, W]
        all_crops.append(crops)
    
    # Stack all crops: [batch_size * 10, 3, H, W]
    batch_crops = torch.cat(all_crops, dim=0)
    
    return batch_crops

def train_fixed_dino(model, train_loader, val_loader, num_epochs=100, lr=1e-4):
    """Train fixed DINO with stable hyperparameters"""
    
    # More conservative optimizer settings
    optimizer = optim.AdamW([
        {'params': model.student_encoder.parameters()},
        {'params': model.student_head.parameters()}
    ], lr=lr, weight_decay=0.04)
    
    # Simple cosine annealing (no warm restarts for stability)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    multi_crop_fn = FixedMultiCropAugmentation(
        global_crops_scale=(0.4, 1.0),
        local_crops_scale=(0.05, 0.4),
        local_crops_number=8,
        global_crops_size=224,
        local_crops_size=96
    )
    
    # Enhanced tracking
    history = {
        'train_loss': [],
        'val_loss': [],  
        'teacher_entropy': [],
        'center_norm': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20  # Increased patience
    
    print("Starting Fixed DINO Training...")
    
    for epoch in range(num_epochs):
        # TRAINING
        model.train()
        train_losses = []
        teacher_entropies = []
        
        # More conservative momentum schedule
        momentum = 0.996 if epoch < 10 else min(0.998, 0.996 + 0.0002 * (epoch - 10))
        
        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            
            # Create multi-crop batch
            crops = create_multi_crop_batch(images, multi_crop_fn)
            crops = crops.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            student_output, teacher_output, _ = model(crops)
            
            # Update center with more conservative momentum
            model.update_center(teacher_output, momentum=0.95)
            
            # Compute loss with conservative temperatures
            loss = fixed_dino_loss(student_output, teacher_output, model.center, 
                                 epoch, num_epochs, student_temp=0.1, teacher_temp=0.07)
            
            loss.backward()
            
            # More aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update teacher
            model.update_teacher(momentum)
            
            # Track metrics
            train_losses.append(loss.item())
            
            # Teacher entropy (should stay high)
            with torch.no_grad():
                teacher_probs = F.softmax((teacher_output - model.center) / 0.07, dim=-1)
                entropy = -(teacher_probs * torch.log(teacher_probs + 1e-8)).sum(dim=-1).mean()
                teacher_entropies.append(entropy.item())
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1:3d}/{num_epochs}, Batch {batch_idx:3d}, '
                      f'Loss: {loss.item():.4f}, Entropy: {entropy.item():.2f}, '
                      f'Center: {torch.norm(model.center).item():.3f}')
        
        # VALIDATION
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                crops = create_multi_crop_batch(images, multi_crop_fn)
                crops = crops.to(device)
                
                student_output, teacher_output, _ = model(crops)
                val_loss = fixed_dino_loss(student_output, teacher_output, model.center,
                                         epoch, num_epochs)
                val_losses.append(val_loss.item())
        
        # Record metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_entropy = np.mean(teacher_entropies)
        center_norm = torch.norm(model.center).item()
        current_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['teacher_entropy'].append(avg_entropy)
        history['center_norm'].append(center_norm)
        history['lr'].append(current_lr)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), './outputs/best_dino_model.pth')
        else:
            patience_counter += 1
        
        print(f'Epoch {epoch+1:3d}/{num_epochs} | Train: {avg_train_loss:.4f} | '
              f'Val: {avg_val_loss:.4f} | Entropy: {avg_entropy:.2f} | '
              f'Center: {center_norm:.3f} | LR: {current_lr:.2e} | '
              f'Momentum: {momentum:.4f}')
        
        # Health checks with warnings
        if avg_entropy < 5.0:
            print("⚠️  WARNING: Teacher entropy is very low - potential mode collapse!")
            print("   Consider reducing learning rate or teacher temperature")
        
        if avg_val_loss > avg_train_loss + 0.5:
            print("⚠️  WARNING: Large train-val gap - potential overfitting!")
            
        # Adaptive early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
            
        # Stop if loss becomes unstable
        if len(history['val_loss']) > 5:
            recent_losses = history['val_loss'][-5:]
            if max(recent_losses) - min(recent_losses) > 2.0:
                print("⚠️  Training became unstable - stopping early")
                break
        
        scheduler.step()
    
    return history

def create_data_loaders(data_dir, batch_size=32, image_size=224, train_split=0.8):
    """Create properly split data loaders"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = CustomImageDataset(data_dir, transform=transform, image_size=image_size)
    
    # Stratified split to ensure balanced classes in train/val
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for label, indices in class_indices.items():
        n_train = int(len(indices) * train_split)
        train_indices.extend(indices[:n_train])
        val_indices.extend(indices[n_train:])
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader, dataset

def plot_fixed_training_progress(history, save_path='./outputs/training_progress.png'):
    """Plot comprehensive training progress"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Validation', linewidth=2)
    axes[0,0].set_title('Loss Curves', fontweight='bold')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Teacher entropy (collapse indicator)
    axes[0,1].plot(epochs, history['teacher_entropy'], 'g-', linewidth=2)
    axes[0,1].axhline(y=6.0, color='red', linestyle='--', alpha=0.7, label='Healthy threshold')
    axes[0,1].set_title('Teacher Entropy\n(Should stay > 6.0)', fontweight='bold')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Entropy')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Center evolution
    axes[0,2].plot(epochs, history['center_norm'], 'purple', linewidth=2)
    axes[0,2].set_title('Center Magnitude Evolution', fontweight='bold')
    axes[0,2].set_xlabel('Epoch')
    axes[0,2].set_ylabel('||Center||')
    axes[0,2].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[1,0].plot(epochs, history['lr'], 'orange', linewidth=2)
    axes[1,0].set_title('Learning Rate Schedule', fontweight='bold')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Learning Rate')
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    
    # Train-val gap
    gap = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1,1].plot(epochs, gap, 'brown', linewidth=2)
    axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1,1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Overfitting threshold')
    axes[1,1].set_title('Generalization Gap\n(Val - Train)', fontweight='bold')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('Loss Difference')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # Training health summary
    axes[1,2].axis('off')
    
    final_entropy = history['teacher_entropy'][-1]
    final_gap = gap[-1]
    min_val_loss = min(history['val_loss'])
    
    health_status = "HEALTHY" if (final_entropy > 6.0 and final_gap < 0.5) else "UNHEALTHY"
    color = "green" if health_status == "HEALTHY" else "red"
    
    summary_text = f"""
TRAINING HEALTH SUMMARY

Status: {health_status}

Final Metrics:
• Teacher Entropy: {final_entropy:.2f}
• Train-Val Gap: {final_gap:.3f}  
• Best Val Loss: {min_val_loss:.4f}

Health Indicators:
• Entropy > 6.0: {'✓' if final_entropy > 6.0 else '✗'}
• Gap < 0.5: {'✓' if final_gap < 0.5 else '✗'}
• Loss Decreasing: {'✓' if history['val_loss'][-1] < history['val_loss'][0] else '✗'}

Expected ARI Range: 0.3 - 0.8
(Negative ARI indicates failure)
"""
    
    axes[1,2].text(0.05, 0.95, summary_text, transform=axes[1,2].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.2))
    
    plt.suptitle('Fixed Multi-Crop DINO: Training Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Fixed training progress saved to: {save_path}")
    plt.show()

def main():
    """Main function with fixed implementation"""
    # Configuration - more conservative settings
    DATA_DIR = "./data/corel"  
    IMAGE_SIZE = 224
    BATCH_SIZE = 8  # Smaller batch size for more stable training
    NUM_EPOCHS = 100
    OUT_DIM = 8192  # Smaller projection dimension
    
    print("="*60)
    print("FIXED MULTI-CROP DINO IMPLEMENTATION")
    print("="*60)
    
    # Load data with proper splitting
    train_loader, val_loader, dataset = create_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE, image_size=IMAGE_SIZE
    )
    
    # Initialize fixed model
    model = FixedMultiCropDINO(input_channels=3, out_dim=OUT_DIM).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train with fixed implementation
    history = train_fixed_dino(model, train_loader, val_loader, num_epochs=NUM_EPOCHS)
    
    # Plot training progress
    plot_fixed_training_progress(history)
    
    # Load best model for evaluation
    if os.path.exists('./outputs/best_dino_model.pth'):
        model.load_state_dict(torch.load('./outputs/best_dino_model.pth'))
        print("Loaded best model weights")
    
    # Visualize multi-crop strategy
    print("Creating multi-crop strategy visualization...")
    visualize_fixed_multicrop_strategy(model, val_loader)
    
    # Evaluate and visualize representations
    print("Evaluating learned representations...")
    features, labels, ari, nmi, silhouette = evaluate_and_visualize_representations(model, val_loader)
    
    print(f"\nFINAL RESULTS:")
    print(f"ARI: {ari:.3f} (should be > 0.0, ideally > 0.3)")
    print(f"NMI: {nmi:.3f}")
    print(f"Silhouette: {silhouette:.3f}")
    
    if ari > 0.0:
        print("SUCCESS: Positive ARI indicates meaningful representations!")
    else:
        print("FAILURE: Negative ARI indicates poor representations")
        print("Try: longer training, different hyperparameters, or more data")

def visualize_fixed_multicrop_strategy(model, val_loader, save_path='./figs/dino_multicrop_examples.png'):
    """Visualize the multi-crop strategy with actual examples"""
    print("Creating multi-crop visualization...")
    
    # Get a batch of images
    for images, labels, _ in val_loader:
        images = images.to(device)
        break
    
    multi_crop_fn = FixedMultiCropAugmentation()
    
    # Create figure for 3 examples
    fig, axes = plt.subplots(3, 11, figsize=(26, 8))
    
    # Denormalization parameters
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    def denormalize(tensor):
        # Ensure mean and std are on the same device as the input tensor
        device = tensor.device
        mean_device = mean.to(device)
        std_device = std.to(device)
        return torch.clamp(tensor * std_device + mean_device, 0, 1)
    
    for sample_idx in range(3):
        img = images[sample_idx]
        label = labels[sample_idx]
        
        # Show original
        img_display = denormalize(img)
        axes[sample_idx, 0].imshow(img_display.permute(1, 2, 0).cpu())
        axes[sample_idx, 0].set_title(f'Original\nClass {label.item()}', fontsize=10, fontweight='bold')
        axes[sample_idx, 0].axis('off')
        axes[sample_idx, 0].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[sample_idx, 0].transAxes,
                                                   fill=False, edgecolor='blue', linewidth=3))
        
        # Generate crops
        crops = multi_crop_fn(img)  # [10, 3, H, W]
        
        # Show 2 global crops (teacher)
        for i in range(2):
            crop_display = denormalize(crops[i])
            axes[sample_idx, i+1].imshow(crop_display.permute(1, 2, 0).cpu())
            axes[sample_idx, i+1].set_title(f'Global {i+1}\n(Teacher)\n224×224', fontsize=9, fontweight='bold')
            axes[sample_idx, i+1].axis('off')
            axes[sample_idx, i+1].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[sample_idx, i+1].transAxes,
                                                         fill=False, edgecolor='green', linewidth=2))
        
        # Show 8 local crops (student) 
        for i in range(8):
            crop_display = denormalize(crops[i+2])
            axes[sample_idx, i+3].imshow(crop_display.permute(1, 2, 0).cpu())
            axes[sample_idx, i+3].set_title(f'Local {i+1}\n(Student)\n96×96', fontsize=8)
            axes[sample_idx, i+3].axis('off')
            axes[sample_idx, i+3].add_patch(plt.Rectangle((0, 0), 1, 1, transform=axes[sample_idx, i+3].transAxes,
                                                         fill=False, edgecolor='orange', linewidth=1))
    
    # Add comprehensive title and information
    fig.suptitle('Fixed Multi-Crop DINO: Local-to-Global Self-Supervised Learning', 
                fontsize=18, fontweight='bold', y=0.95)
    
    # Add legend and information
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='blue', linewidth=3, label='Original Image'),
        Patch(facecolor='white', edgecolor='green', linewidth=2, label='Global Crops → Teacher Network'),
        Patch(facecolor='white', edgecolor='orange', linewidth=1, label='Local Crops → Student Network')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=14, bbox_to_anchor=(0.5, 0.02))
    
    # Add training strategy explanation
    strategy_text = """
DINO Training Strategy:
• Student (local crops) learns to predict Teacher (global crops)
• Teacher updated via EMA from Student
• Asymmetric loss: 8 local × 2 global = 16 prediction pairs
• Centering prevents mode collapse
"""
    fig.text(0.02, 0.02, strategy_text, fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    
    # Save with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Multi-crop visualization saved to: {save_path}")
    plt.show()

def evaluate_and_visualize_representations(model, val_loader, save_path='./figs/dino_multicrop_representations.png'):
    """Complete evaluation with UMAP visualization"""
    print("Extracting learned representations...")
    
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images = images.to(device)
            features = model.encode_images(images)
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    features = np.vstack(all_features)
    labels = np.array(all_labels)
    
    print(f"Extracted features shape: {features.shape}")
    print(f"Number of classes: {len(np.unique(labels))}")
    
    # PCA preprocessing
    features_pca = features
    if features.shape[1] > 50:
        pca = PCA(n_components=50)
        features_pca = pca.fit_transform(features)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"PCA explained variance: {explained_var:.3f}")
    
    # UMAP projection
    print("Computing UMAP projection...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    features_umap = umap_reducer.fit_transform(features_pca)
    
    # Clustering evaluation
    n_clusters = len(np.unique(labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(features_pca)
    
    ari = adjusted_rand_score(labels, pred_labels)
    nmi = normalized_mutual_info_score(labels, pred_labels)
    silhouette = silhouette_score(features_pca, labels)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Main UMAP plot
    scatter = axes[0,0].scatter(features_umap[:, 0], features_umap[:, 1], c=labels, 
                               cmap='tab10', alpha=0.7, s=40)
    axes[0,0].set_title(f'UMAP Projection of DINO Features\nARI: {ari:.3f} | NMI: {nmi:.3f} | Silhouette: {silhouette:.3f}', 
                       fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('UMAP Component 1')
    axes[0,0].set_ylabel('UMAP Component 2')
    axes[0,0].grid(True, alpha=0.3)
    
    # Class distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    axes[0,1].bar(unique_labels, counts, color='lightblue', alpha=0.7)
    axes[0,1].set_xlabel('Class Label')
    axes[0,1].set_ylabel('Number of Samples')
    axes[0,1].set_title('Class Distribution', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Clustering quality comparison
    methods = ['ARI', 'NMI', 'Silhouette']
    scores = [ari, nmi, silhouette]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    
    bars = axes[1,0].bar(methods, scores, color=colors, alpha=0.8)
    axes[1,0].set_ylabel('Score')
    axes[1,0].set_title('Clustering Quality Metrics', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_ylim(-0.1, 1.0)
    
    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                      f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Results interpretation
    axes[1,1].axis('off')
    
    # Determine result quality
    if ari > 0.5:
        quality = "EXCELLENT"
        color = "green"
    elif ari > 0.2:
        quality = "GOOD" 
        color = "orange"
    elif ari > 0.0:
        quality = "FAIR"
        color = "yellow"
    else:
        quality = "POOR"
        color = "red"
    
    results_text = f"""
REPRESENTATION LEARNING RESULTS

Overall Quality: {quality}

Metrics Interpretation:
• ARI = {ari:.3f}
  {"> 0.5: Excellent" if ari > 0.5 else "> 0.2: Good" if ari > 0.2 else "> 0.0: Fair" if ari > 0.0 else "< 0.0: Poor (Failed)"}

• NMI = {nmi:.3f}
  {"> 0.7: Excellent" if nmi > 0.7 else "> 0.5: Good" if nmi > 0.5 else "> 0.3: Fair" if nmi > 0.3 else "< 0.3: Poor"}

• Silhouette = {silhouette:.3f}
  {"> 0.5: Excellent" if silhouette > 0.5 else "> 0.3: Good" if silhouette > 0.3 else "> 0.1: Fair" if silhouette > 0.1 else "< 0.1: Poor"}

Dataset: {len(features)} samples, {len(np.unique(labels))} classes
Feature dim: {features.shape[1]} → {features_pca.shape[1]} (PCA)

{'✓ SUCCESS: DINO learned meaningful representations' if ari > 0.0 else '✗ FAILURE: Training did not converge properly'}
"""
    
    axes[1,1].text(0.05, 0.95, results_text, transform=axes[1,1].transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.2))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes, shrink=0.6, aspect=20)
    cbar.set_label('Class Labels', fontsize=12)
    
    plt.suptitle('Fixed Multi-Crop DINO: Representation Learning Evaluation', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, right=0.85)
    
    # Save visualization
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"UMAP visualization saved to: {save_path}")
    plt.show()
    
    return features, labels, ari, nmi, silhouette

if __name__ == "__main__":
    main()
