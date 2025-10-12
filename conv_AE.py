import os
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch_snippets.torch_loader import Report

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
# ConvAutoEncoder
# ------------------------------

class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 5, stride=3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 2, stride=2, padding=1), nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)
        return x
    
    def encode_only(self, x):
        return self.encoder(x)
    
# ------------------------------
# Training & Validation Batches
# ------------------------------

def train_batch(batch, model, criterion, optimizer, device="cuda"):
    model.train()
    xb, _, _ = batch
    xb = xb.to(device)
    optimizer.zero_grad()
    out = model(xb)
    loss = criterion(out, xb)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def validate_batch(batch, model, criterion, device="cuda"):
    model.eval()
    xb, _, _ = batch
    xb = xb.to(device)
    out = model(xb)
    loss = criterion(out, xb)
    return loss.item()

# ------------------------------
# Feature Extraction
# ------------------------------

def show_reconstructions(model, dataset, device="cuda", n_samples=5):
    import matplotlib.pyplot as plt
    model.eval()
    for i in range(n_samples):
        ix = np.random.randint(len(dataset))
        im, _, _ = dataset[ix]
        _im = model(im[None].to(device))[0].detach().cpu()
        fig, ax = plt.subplots(1,2, figsize=(6,3))
        ax[0].imshow(im.permute(1,2,0))
        ax[0].set_title("Input")
        ax[1].imshow(_im.permute(1,2,0))
        ax[1].set_title("Reconstruction")
        plt.show()

        fig.savefig(f"./figs/conv autoencoder/reconstruction_{i}.png")


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
    fig.savefig("./figs/conv autoencoder/conv_autoencoder_umap_comparison.png")

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
DATA_DIR = "./data/corel"
BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_EPOCHS = 22

trn_dl, val_dl, dataset = create_data_loaders(DATA_DIR, BATCH_SIZE, IMAGE_SIZE)

model = ConvAutoEncoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

log = Report(NUM_EPOCHS)


# Treinamento rápido
for epoch in range(NUM_EPOCHS):
    epoch_losses = []
    N = len(trn_dl)

    for batch in trn_dl:
        loss = train_batch(batch, model, criterion, optimizer, device)
        epoch_losses.append(loss)
        log.record(pos=(epoch+1)/NUM_EPOCHS, train_loss=loss, end="\r")


    val_losses = []
    N=len(val_dl)
    for batch in val_dl:
        loss = validate_batch(batch, model, criterion, device)
        val_losses.append(loss)
        log.record(pos=(epoch+1)/NUM_EPOCHS, val_loss=loss, end="\r")

    print(f"Epoch {epoch+1}/{NUM_EPOCHS},Val Loss: {np.mean(val_losses):.4f}")


print("Convolutional Autoencoder Training completed!")

# Cria a figura e o eixo
fig, ax = plt.subplots(figsize=(8,6))

# Gera o gráfico usando seu método
log.plot_epochs(log=True, ax=ax)  # assume que plot_epochs aceita parâmetro 'ax'

# Ajustes opcionais
ax.set_title("Conv Autoencoder- Model Loss", fontsize=14, fontweight='bold')
ax.set_xlabel("Epoch")
ax.set_ylabel("Metrics")

# Salva a figura
plt.tight_layout()
plt.savefig("./figs/conv autoencoder/loss.png", dpi=300, bbox_inches='tight')

# Mostra a figura
plt.show()


log.plot_epochs(log=True)


# Visualizar reconstruções
show_reconstructions(model, dataset, device, n_samples=3)

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

export_text="Convolutional AutoEncoder Clustering Metrics\n\n"
for method, m in metrics.items():
    text=f"{method}: Silhouette={m['silhouette']:.3f}, ARI={m['ari']:.3f}"
    print(text)
    export_text+=text+"\n"

with open("./figs/conv autoencoder/metrics.txt", "w") as f:
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
plt.title('t-SNE Projection of Conv Autoencoder Latent Space', fontsize=14, fontweight='bold')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.savefig('./figs/conv autoencoder/conv_autoencoder_tsne.png', dpi=300, bbox_inches='tight')
plt.show()

# ------------------------------
# Image Generation
# ------------------------------

# Extrair vetores latentes
latent_vectors = []
classes = []
model.eval()
with torch.no_grad():
    for im, clss, _ in val_dl:
        im = im.to(device)
        z = model.encoder(im)
        latent_vectors.append(z)
        classes.extend(clss)

latent_vectors = torch.cat(latent_vectors)  # shape: [N, latent_dim_channels, h, w]
latent_vectors_flat = latent_vectors.view(latent_vectors.size(0), -1)

# Geração aleatória
n_samples = 10  # para preencher 10x10 subplots
rand_vectors = []

latent_vectors_flat = latent_vectors.view(latent_vectors.size(0), -1)

for col in latent_vectors_flat.T:
    mu, sigma = col.mean(), col.std()
    rand_vectors.append(mu + sigma*torch.randn(n_samples, device=device))

rand_vectors = torch.stack(rand_vectors, dim=1)  # shape: [100, latent_dim_flat]

# Decodificação
fig, axes = plt.subplots(2, 5, figsize=(7,7))
axes = axes.flatten()

for i, p in enumerate(rand_vectors):
    p_reshaped = p.view(1, latent_vectors.size(1), latent_vectors.size(2), latent_vectors.size(3))
    img = model.decoder(p_reshaped).squeeze().detach().cpu().numpy()

    # Se RGB, transpor de [C,H,W] para [H,W,C]
    if img.ndim == 3:
        img = np.transpose(img, (1,2,0))
        
        
    img = (img - img.min()) / (img.max() - img.min())

    axes[i].imshow(img, cmap='gray' if img.ndim==2 else None)
    axes[i].axis('off')

plt.suptitle('Random Generated Images from Latent Space', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('./figs/conv autoencoder/conv_autoencoder_random_generation.png', dpi=300, bbox_inches='tight')
plt.show()










