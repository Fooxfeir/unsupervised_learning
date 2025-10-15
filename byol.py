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
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader, dataset


# ------------------------------
# Byol Model
# ------------------------------
def create_encoder(input_channels=3):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, 3, stride=3, padding=1), 
        nn.BatchNorm2d(32),
        nn.ReLU(True),
        nn.MaxPool2d(2, stride=2),
        nn.Conv2d(32, 64, 3, stride=2, padding=1), 
        nn.BatchNorm2d(64),
        nn.ReLU(True),
        nn.MaxPool2d(2, stride=1)
    )

def create_projector(feature_dim, projection_dim=64):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(feature_dim, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(True),
        nn.Linear(256, projection_dim)
    )

def create_predictor(projection_dim=64):
    return nn.Sequential(
        nn.Linear(projection_dim, projection_dim//2),
        nn.BatchNorm1d(projection_dim//2),
        nn.ReLU(True),
        nn.Linear(projection_dim//2, projection_dim)
    )


class BYOLModel(nn.Module):
    def __init__(self, input_channels=3, projection_dim=64, input_size=(3,224,224)):
        super().__init__()

        # Determinar feature_dim dinamicamente
        with torch.no_grad():
            dummy = torch.zeros(1, *input_size)
            feat = create_encoder(input_channels)(dummy)
            self.feature_dim = feat.view(1,-1).size(1)

        # Online network
        self.online_encoder = create_encoder(input_channels)
        self.online_projector = create_projector(self.feature_dim, projection_dim)
        self.predictor = create_predictor(projection_dim)

        # Target network
        self.target_encoder = create_encoder(input_channels)
        self.target_projector = create_projector(self.feature_dim, projection_dim)
        self._initialize_target_network()

        # Freeze target
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def _initialize_target_network(self):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data.copy_(online_param.data)
        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data.copy_(online_param.data)

    def update_target_network(self, momentum=0.996):
        for online_param, target_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = momentum*target_param.data + (1-momentum)*online_param.data
        for online_param, target_param in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_param.data = momentum*target_param.data + (1-momentum)*online_param.data

    def forward(self, x1, x2):
        online_f1 = self.online_encoder(x1)
        online_f2 = self.online_encoder(x2)
        online_p1 = self.online_projector(online_f1)
        online_p2 = self.online_projector(online_f2)
        pred1 = self.predictor(online_p1)
        pred2 = self.predictor(online_p2)

        with torch.no_grad():
            target_f1 = self.target_encoder(x1)
            target_f2 = self.target_encoder(x2)
            target_p1 = self.target_projector(target_f1)
            target_p2 = self.target_projector(target_f2)

        return pred1, pred2, target_p1, target_p2
    

    def encode_only(self, x):
        """Retorna apenas o embedding do encoder online, sem projector/predictor"""
        with torch.no_grad():
            feat = self.online_encoder(x)
        return feat
    

def byol_loss(online_pred1, online_pred2, target_proj1, target_proj2):
    online_pred1 = F.normalize(online_pred1, dim=1)
    online_pred2 = F.normalize(online_pred2, dim=1)
    target_proj1 = F.normalize(target_proj1, dim=1)
    target_proj2 = F.normalize(target_proj2, dim=1)
    
    loss1 = 2 - 2 * (online_pred1 * target_proj2).sum(dim=1).mean()
    loss2 = 2 - 2 * (online_pred2 * target_proj1).sum(dim=1).mean()
    
    return (loss1 + loss2) / 2

def to_tensor_rgb(img):
    if isinstance(img, np.ndarray):
        img_tensor = torch.tensor(img).permute(2,0,1).float()  # [C,H,W]
    elif isinstance(img, torch.Tensor) and img.ndim==3:
        img_tensor = img.permute(2,0,1).float()
    else:
        raise ValueError("Imagem deve ser np.ndarray ou tensor 3D")
    return img_tensor

def create_byol_pairs(batch, augment_fn):
    """
    batch: tensor [B, C, H, W] com valores normalizados [-1,1]
    retorna: aug1, aug2 [B, C, H, W]
    """
    batch_denorm = (batch * 0.5 + 0.5).clamp(0,1)  # [-1,1] -> [0,1]
    aug1_list, aug2_list = [], []

    for img in batch_denorm:
        # img: [C,H,W] -> [H,W,C] e np.array float32
        img_np = img.permute(1,2,0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)  # To uint8 for PIL
        aug1 = augment_fn(img_np)
        aug2 = augment_fn(img_np)
        # PIL -> tensor [C,H,W] float [0,1]
        aug1_list.append(transforms.ToTensor()(aug1))
        aug2_list.append(transforms.ToTensor()(aug2))

    return torch.stack(aug1_list), torch.stack(aug2_list)



def get_byol_augmentation():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Lambda(lambda x: np.array(x)),  # mantém como ndarray para ToTensor posterior
        transforms.Lambda(lambda x: x + 5*np.random.randn(*x.shape).astype(np.float32)),  # small noise
        transforms.Lambda(lambda x: np.clip(x, 0, 255))
    ])


# ------------------------------
# Training & Validation Batches
# ------------------------------


def train_batch_byol(data, model, optimizer, augment_fn, momentum=0.996, device='cuda'):
    model.train()
    data = data.to(device)
    aug1, aug2 = create_byol_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    
    optimizer.zero_grad()
    pred1, pred2, target1, target2 = model(aug1, aug2)
    loss = byol_loss(pred1, pred2, target1, target2)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    model.update_target_network(momentum)
    return loss

@torch.no_grad()
def validate_batch_byol(data, model, augment_fn, device='cuda'):
    model.eval()
    data = data.to(device)
    aug1, aug2 = create_byol_pairs(data, augment_fn)
    aug1, aug2 = aug1.to(device), aug2.to(device)
    pred1, pred2, target1, target2 = model(aug1, aug2)
    loss = byol_loss(pred1, pred2, target1, target2)
    return loss

# ------------------------------
# Feature Extraction
# ------------------------------


def extract_data_for_comparison(model, dataloader, max_samples=2000):
    """Extract raw data, latent features, and labels safely, ignoring extra batch elements"""
    model.eval()
    
    raw_data = []
    latent_features = []
    labels = []
    
    print("Extracting data for UMAP comparison...")
    
    with torch.no_grad():
        samples_collected = 0
        for batch in dataloader:
            if samples_collected >= max_samples:
                break
            
            # Pega apenas os dois primeiros elementos do batch
            data, label = batch[:2]
            
            data = data.to(device)
            
            # Raw pixel data
            raw_pixels = data.view(data.size(0), -1).cpu().numpy()
            raw_data.append(raw_pixels)
            
            # Latent features from online encoder (without projector/predictor)
            latent = model.encode_only(data)
            latent_flat = latent.view(latent.size(0), -1).cpu().numpy()
            latent_features.append(latent_flat)
            
            # Labels
            labels.append(label.numpy())
            
            samples_collected += len(data)
    
    raw_data = np.vstack(raw_data)[:max_samples]
    latent_features = np.vstack(latent_features)[:max_samples]
    labels = np.hstack(labels)[:max_samples]
    
    print(f"Extracted {len(raw_data)} samples:")
    print(f"  Raw data shape: {raw_data.shape}")
    print(f"  Latent features shape: {latent_features.shape}")
    
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
    fig.savefig("./figs/byol/conv_autoencoder_umap_comparison.png")

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


DATA_DIR = "/home/guilhermo.oliveira/unsupervised_learning/UnsupervisedFeatureLearningCNNs/data/corel"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_EPOCHS = 5

trn_dl, val_dl, dataset = create_data_loaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)

model = BYOLModel(input_channels=3, projection_dim=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

augment_fn = get_byol_augmentation()

num_epochs = 22 # Reduced epochs for MNIST
log = Report(num_epochs)

for epoch in range(num_epochs):
    
    epoch_losses = []
    N = len(trn_dl)

    for batch in trn_dl:
        loss=train_batch_byol(batch[0], model, optimizer, augment_fn)
        epoch_losses.append(loss.item())
        log.record(pos=(epoch + (len(epoch_losses))/N), loss=loss.item(), end="\r")

    val_losses = []
    N=len(val_dl)

    for batch in val_dl:
        loss=validate_batch_byol(batch[0], model, augment_fn)
        val_losses.append(loss.item())
        log.record(pos=(epoch + (len(val_losses))/N), val_loss=loss.item(), end="\r")
    
    
    print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {np.mean(val_losses):.4f}")

    scheduler.step()



print("Byol Training completed!")

# Cria a figura e o eixo
fig, ax = plt.subplots(figsize=(8,6))

# Gera o gráfico usando seu método
log.plot_epochs(log=True, ax=ax)  # assume que plot_epochs aceita parâmetro 'ax'

# Ajustes opcionais
ax.set_title("Byol - Model Loss ", fontsize=14, fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Metrics")

# Salva a figura
plt.tight_layout()
plt.savefig("./figs/byol/loss.png", dpi=300, bbox_inches='tight')

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
original = (sample_img[0] * 0.5 + 0.5).clamp(0, 1)
axes[0].imshow(original.permute(1, 2, 0).cpu().numpy())
axes[0].set_title('Original')
axes[0].axis('off')

# ===== 9 versões aumentadas =====
for i in range(1, 10):
    # Converter para numpy uint8 [H,W,C] para o augment_fn
    img_np = (original.permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
    
    # Aplicar augmentação
    aug_img = augment_fn(img_np)
    
    # Converter de volta para tensor [C,H,W] float [0,1] para exibição
    aug_tensor = transforms.ToTensor()(aug_img).clamp(0, 1)

    axes[i].imshow(aug_tensor.permute(1, 2, 0).cpu().numpy())
    axes[i].set_title(f'Aug {i}')
    axes[i].axis('off')

plt.suptitle('BYOL Data Augmentations (RGB)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/byol/simclr_augmentations.png', dpi=300, bbox_inches='tight')
plt.show()





# Extrair features e aplicar PCA+UMAP
raw_data, latent_features, labels = extract_data_for_comparison(model, val_dl, max_samples=2000)
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

with open("./figs/byol/metrics.txt", "w") as f:
    f.write(export_text)

# ------------------------------
# T-Sne Plot
# ------------------------------

device = next(model.parameters()).device  # pega a device do modelo

latent_vectors = []
classes = []

model.eval()

for im, clss, _ in val_dl:
    im = im.to(device)
    latent_vectors.append(model.encode_only(im).view(len(im), -1))
    classes.append(clss)

latent_vectors = torch.cat(latent_vectors).cpu().detach().numpy()
classes = torch.cat(classes).cpu().numpy()

tsne = TSNE(n_components=2, random_state=42)
clustered = tsne.fit_transform(latent_vectors)

fig = plt.figure(figsize=(12,10))
cmap = plt.get_cmap('Spectral', 10)
plt.scatter(clustered[:,0], clustered[:,1], c=classes, cmap=cmap)
plt.colorbar(ticks=range(10))
plt.title('t-SNE Projection of BYOL Latent Space', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('./figs/byol/conv_autoencoder_tsne.png', dpi=300, bbox_inches='tight')
plt.show()
