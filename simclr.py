import torch
from torch import nn as nn
from torch.utils.data import TensorDataset, DataLoader,Dataset
from torch_snippets import *
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_snippets.torch_loader import Report
import umap
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import os
import glob
from PIL import Image

# ------------------------------
# Custom Dataset
# ------------------------------

class CustomImageDataset(Dataset):
    def __init__(self, data_dir, transform=None, image_size=224):
        self.data_dir = data_dir
        self.transform = transform
        self.image_size = image_size
        
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        self.image_paths.sort()
        if len(self.image_paths) == 0:
            raise ValueError(f"No PNG files found in {data_dir}")
        
        self.labels = []
        for path in self.image_paths:
            filename = os.path.basename(path)
            try:
                class_num = int(filename.split('_')[0])
                self.labels.append(class_num - 1)
            except:
                self.labels.append(hash(filename.split('.')[0]) % 10)
        
        print(f"Found {len(self.image_paths)} images, classes: {sorted(set(self.labels))}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # RGB
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label, idx


# ------------------------------
# Data Loaders
# ------------------------------


def create_data_loaders(data_dir, batch_size=32, image_size=224, train_split=0.8):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    
    dataset = CustomImageDataset(data_dir, transform=transform, image_size=image_size)
    
    # Stratified split
    from collections import defaultdict
    class_indices = defaultdict(list)
    for idx, label in enumerate(dataset.labels):
        class_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    for label, indices in class_indices.items():
        n_train = int(len(indices)*train_split)
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


# ------------------------------
# SimCLR Model
# ------------------------------
class SimCLREncoder(nn.Module):
    def __init__(self, latent_dim=128, input_shape=(3, 224, 224)):
        super().__init__()
        # Encoder CNN
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )

        # calcular feature_dim automaticamente
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)  # (1,3,224,224)
            out = self.encoder(dummy)
            self.feature_dim = out.numel()

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection_head(features)
        return features, projections
    
    def encode_only(self, x):
        return self.encoder(x)
    
def get_simclr_augmentation():
    """Augmentation pipeline for SimCLR (RGB)"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + 0.05 * torch.randn_like(x)),  # Noise
        transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalização RGB
    ])

def create_simclr_pairs(batch, augment_fn):
    """Create augmented pairs for SimCLR training (RGB)"""
    batch_size = batch.shape[0]
    
    augmented_1, augmented_2 = [], []
    
    for img in batch:
        img_denorm = (img * 0.5 + 0.5)  # voltar de [-1,1] para [0,1]
        
        aug1 = augment_fn(img_denorm)
        aug2 = augment_fn(img_denorm)
        
        augmented_1.append(aug1)
        augmented_2.append(aug2)
    
    return torch.stack(augmented_1), torch.stack(augmented_2)
    
def simclr_loss(projections_1, projections_2, temperature=0.5, device="cuda"):
    batch_size = projections_1.shape[0]
    
    projections_1 = F.normalize(projections_1, dim=1)
    projections_2 = F.normalize(projections_2, dim=1)
    
    projections = torch.cat([projections_1, projections_2], dim=0)
    similarity_matrix = torch.matmul(projections, projections.T) / temperature
    
    labels = torch.cat([torch.arange(batch_size, 2*batch_size), 
                        torch.arange(0, batch_size)]).to(device)
    
    mask = torch.eye(2*batch_size).bool().to(device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -9e15)
    
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss

# ------------------------------
# Training & Validation Batches
# ------------------------------


def train_batch_simclr(data, model, optimizer, augment_fn, temperature=0.5):
    model.train()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_simclr_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    optimizer.zero_grad()
    
    # Forward pass through both augmented versions
    _, proj1 = model(aug1)
    _, proj2 = model(aug2)
    
    # Compute SimCLR loss
    loss = simclr_loss(proj1, proj2, temperature)
    
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    return loss

@torch.no_grad()
def validate_batch_simclr(data, model, augment_fn, temperature=0.5):
    model.eval()
    data = data.to(device)
    
    # Create augmented pairs
    aug1, aug2 = create_simclr_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    # Forward pass
    _, proj1 = model(aug1)
    _, proj2 = model(aug2)
    
    # Compute loss
    loss = simclr_loss(proj1, proj2, temperature)
    
    return loss


# ------------------------------
# Feature Extraction
# ------------------------------


def extract_data_for_comparison(model, dataloader, max_samples=2000, device="cuda"):
    model.eval()
    raw_data, latent_features, labels = [], [], []
    with torch.no_grad():
        samples_collected = 0
        for batch in dataloader:
            images, batch_labels, _ = batch
            if samples_collected >= max_samples:
                break
            images = images.to(device)
            raw_data.append(images.view(images.size(0), -1).cpu().numpy())
            latent = model.encode_only(images)
            latent_features.append(latent.view(latent.size(0), -1).cpu().numpy())
            labels.append(batch_labels.numpy())
            samples_collected += images.size(0)
    
    raw_data = np.vstack(raw_data)[:max_samples]
    latent_features = np.vstack(latent_features)[:max_samples]
    labels = np.hstack(labels)[:max_samples]
    return raw_data, latent_features, labels


# ------------------------------
# PCA Reduction
# ------------------------------

def apply_pca_reduction(data, n_components):
    n_components = min(n_components, data.shape[0], data.shape[1])
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=n_components, random_state=42)
    data_pca = pca.fit_transform(data_scaled)
    return data_pca


# ------------------------------
# UMAP Projection
# ------------------------------

def apply_umap_projection(data, title, n_neighbors=15, min_dist=0.1, random_state=42):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(data)
    return embedding

def plot_umap_comparison(embeddings_dict, labels):
    fig, axes = plt.subplots(1, len(embeddings_dict), figsize=(5*len(embeddings_dict), 4))
    if len(embeddings_dict)==1: axes=[axes]
    for ax, (title, embedding) in zip(axes, embeddings_dict.items()):
        ax.scatter(embedding[:,0], embedding[:,1], c=labels, cmap="tab10", s=5)
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
    fig.savefig("./figs/simclr/conv_autoencoder_umap_comparison.png")

# ------------------------------
# Clustering Metrics
# ------------------------------


def calculate_metrics(embedding, labels, n_clusters=None):
    if n_clusters is None:
        n_clusters = len(np.unique(labels))
    # Rodar KMeans no embedding
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    pred_labels = kmeans.fit_predict(embedding)
    
    sil = silhouette_score(embedding, pred_labels)
    ari = adjusted_rand_score(labels, pred_labels)
    return {"silhouette": sil, "ari": ari}


# ------------------------------
# MAIN 
# ------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


DATA_DIR = "./data/corel"
BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_EPOCHS = 5

trn_dl, val_dl, dataset = create_data_loaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)

model = SimCLREncoder(latent_dim=128).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)  # Increased LR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)  # Simpler scheduler

augment_fn = get_simclr_augmentation()

num_epochs = 22 # Reduced epochs for MNIST
log = Report(num_epochs)

for epoch in range(num_epochs):
    
    epoch_losses = []
    N = len(trn_dl)

    for batch in trn_dl:
        loss=train_batch_simclr(batch[0], model, optimizer, augment_fn)
        epoch_losses.append(loss.item())
        log.record(pos=(epoch + (len(epoch_losses))/N), loss=loss.item(), end="\r")

    val_losses = []
    N=len(val_dl)

    for batch in val_dl:
        loss=validate_batch_simclr(batch[0], model, augment_fn)
        val_losses.append(loss.item())
        log.record(pos=(epoch + (len(val_losses))/N), val_loss=loss.item(), end="\r")
    
    
    print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {np.mean(val_losses):.4f}")

    scheduler.step()


print("SimCLR Training completed!")

# Cria a figura e o eixo
fig, ax = plt.subplots(figsize=(8,6))

# Gera o gráfico usando seu método
log.plot_epochs(log=True, ax=ax)  # assume que plot_epochs aceita parâmetro 'ax'

# Ajustes opcionais
ax.set_title("SimCLR - Model Loss", fontsize=14, fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Metrics")

# Salva a figura
plt.tight_layout()
plt.savefig("./figs/simclr/loss.png", dpi=300, bbox_inches='tight')

# Mostra a figura
plt.show()


log.plot_epochs(log=True)

# ------------------------------
# Data Augmentations
# ------------------------------

sample_data, sample_labels, sample_indices = next(iter(val_dl))
sample_img = sample_data[0:1]  # pega só a primeira imagem do batch

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
axes = axes.ravel()

# ===== Original image =====
# Denormalizar de [-1,1] para [0,1] e converter [C,H,W] -> [H,W,C]
original = (sample_img[0].permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)
axes[0].imshow(original.numpy())
axes[0].set_title('Original')
axes[0].axis('off')

# ===== 9 versões aumentadas =====
for i in range(1, 10):
    # Denormalizar antes da augmentação
    img_for_aug = (sample_img[0] * 0.5 + 0.5)  # [3,H,W] em [0,1]
    aug_img = augment_fn(img_for_aug)          # aplica augment_fn (SimCLR)

    # Denormalizar de volta para exibição
    aug_display = (aug_img.permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1)

    axes[i].imshow(aug_display.numpy())
    axes[i].set_title(f'Aug {i}')
    axes[i].axis('off')

plt.suptitle('SimCLR Data Augmentations (RGB)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/simclr/simclr_augmentations.png', dpi=300, bbox_inches='tight')
plt.show()




# Extrair features e aplicar PCA+UMAP
raw_data, latent_features, labels = extract_data_for_comparison(model, val_dl, max_samples=2000, device=device)
latent_dim = latent_features.shape[1]
pca_features = apply_pca_reduction(latent_features, n_components=50)

embeddings = {
    'Original Data': apply_umap_projection(raw_data, 'Original Data'),
    f'Conv Encoder Latent ({latent_dim}D)': apply_umap_projection(latent_features, 'Latent Features'),
    f'PCA Reduced ({pca_features.shape[1]}D)': apply_umap_projection(pca_features, 'PCA')
}

plot_umap_comparison(embeddings, labels)

# ------------------------------
# Metrics
# ------------------------------
metrics = {k: calculate_metrics(v, labels) for k,v in embeddings.items()}

export_text="SimCLR Clustering Metrics\n\n"
for method, m in metrics.items():
    text=f"{method}: Silhouette={m['silhouette']:.3f}, ARI={m['ari']:.3f}"
    print(text)
    export_text+=text+"\n"

with open("./figs/simclr/metrics.txt", "w") as f:
    f.write(export_text)

# ------------------------------
# T-Sne Plot
# ------------------------------

device = next(model.parameters()).device  # pega a device do modelo

latent_vectors = []
classes = []

model.eval()
with torch.no_grad():
    for im, clss, _ in val_dl:
        im = im.to(device)       # move input para a mesma device do modelo
        clss = clss.to(device)   # se precisar das classes no mesmo device

        z = model.encoder(im)
        z = z.view(z.size(0), -1)
        
        latent_vectors.append(z.cpu())  # movemos para CPU para armazenar
        classes.append(clss.cpu())

latent_vectors = torch.cat(latent_vectors).numpy()
classes = torch.cat(classes).numpy()

tsne = TSNE(2)
clustered = tsne.fit_transform(latent_vectors)

fig = plt.figure(figsize=(12,10))
cmap = plt.get_cmap('Spectral', 10)
plt.scatter(*zip(*clustered), c=classes, cmap=cmap)
plt.colorbar(drawedges=True)
plt.title('t-SNE Projection of SimCLR Latent Space', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('./figs/simclr/conv_autoencoder_tsne.png', dpi=300, bbox_inches='tight')
plt.show()