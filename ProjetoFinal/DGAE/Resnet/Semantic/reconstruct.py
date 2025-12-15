import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from diffusers import DDPMScheduler
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.auto import tqdm
import math
import numpy as np
import random
import warnings

# Imports para Análise de Clusterização
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap 
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

warnings.filterwarnings("ignore")

# --- CONFIGURAÇÕES ---
MODEL_TYPE = "Semantic"  # Agora focado no vetor achatado
Augmented = False         # Ajuste conforme onde treinou

# Definição de Caminhos
if MODEL_TYPE == "Semantic":
    if Augmented:
        # Nome do arquivo salvo no treino Vector VAE (ajuste se necessário)
        MODEL_PATH = "dgae_semantic_aug.pth" 
        DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel augmented"
        out_prefix = "semantic_aug"
        title = "DGAE Vector 1D (Augmented)"
    else:
        MODEL_PATH = "dgae_semantic_base.pth"
        DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel"
        out_prefix = "semantic_base"
        title = "DGAE Vector 1D (Base)"
    
    # CONFIGURAÇÃO DE VETOR
    LATENT_DIM = 64      # Vetor 1D
    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    NUM_TIMESTEPS = 1000 
    NUM_RECONSTRUCTIONS = 4 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. ARQUITETURAS SEMANTIC (VECTOR / FLATTENED)
# ==============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

# --- ENCODER GLOBAL (Achatamento) ---
class Semantic_VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        resnet = models.resnet18(weights=None)
        
        # Removemos FC e AvgPool originais para controlar o fluxo
        self.features = nn.Sequential(*list(resnet.children())[:-2]) 
        
        # Global Average Pooling: [B, 512, H, W] -> [B, 512, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # Duas cabeças lineares para VAE (Média e Variância)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = self.flatten(x)       # [B, 512]
        
        mu = self.fc_mu(x)        # [B, 64]
        log_var = self.fc_var(x)  # [B, 64]
        
        return mu, log_var

# --- DECODER GLOBAL (Expandindo Vetor) ---
class Semantic_Diffusion_Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(128), nn.Linear(128, 128), nn.GELU())
        
        # Input: 3 canais imagem + latent_dim (expandido como canais)
        in_channels = 3 + latent_dim 
        
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU()) 
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(8, 256), nn.GELU()) 
        self.bottleneck = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.GELU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU()) 
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU())  
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x, t, z_vector):
        # Expansão espacial do vetor para concatenar com a imagem
        # z_vector: [B, 64] -> [B, 64, 1, 1]
        z_spatial = z_vector.view(z_vector.shape[0], z_vector.shape[1], 1, 1)
        # Expande para [B, 64, H, W] para bater com a imagem ruidosa
        z_expanded = z_spatial.expand(-1, -1, x.shape[2], x.shape[3])
        
        x_input = torch.cat([x, z_expanded], dim=1)
        
        t_emb = self.time_mlp(t)[..., None, None]
        x1 = self.down1(x_input)
        x2 = self.down2(x1) + t_emb 
        x3 = self.down3(x2)
        mid = self.bottleneck(x3)
        return self.out(self.up2(self.up1(mid + x3) + x2) + x1)

class DGAE_Semantic(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = Semantic_VAE_Encoder(latent_dim)
        self.decoder = Semantic_Diffusion_Decoder(latent_dim)

# ==============================================================================
# 2. DATASET
# ==============================================================================
class CorelDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.files = sorted(glob.glob(os.path.join(data_dir, "**", "*.png"), recursive=True) + 
                            glob.glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
        
        self.labels = []
        for p in self.files:
            filename = os.path.basename(p)
            try:
                cls_str = filename.split('_')[0]
                self.labels.append(cls_str)
            except:
                self.labels.append("unknown")
        
        unique_classes = sorted(list(set(self.labels)))
        self.class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        
        print(f"Dataset Info: {len(self.files)} imagens. Classes: {unique_classes}")

    def __len__(self): return len(self.files)
    
    def __getitem__(self, idx): 
        img_path = self.files[idx]
        image = self.transform(Image.open(img_path).convert('RGB'))
        label_str = self.labels[idx]
        label_idx = self.class_to_idx[label_str]
        return image, label_idx

def unnormalize(tensor):
    return (tensor * 0.5 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()

# ==============================================================================
# 3. RECONSTRUÇÃO
# ==============================================================================
def gerar_reconstrucoes(model, dataset):
    print("\n--- Gerando Reconstruções ---")
    model.eval()
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    
    n_samples = min(NUM_RECONSTRUCTIONS, len(dataset))
    indices = random.sample(range(len(dataset)), n_samples)
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4 * n_samples, 6))
    if n_samples == 1: axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        data = dataset[idx]
        original_img = data[0].unsqueeze(0).to(device)
        
        label_idx = data[1]
        if isinstance(label_idx, torch.Tensor): label_idx = label_idx.item()
        class_name = dataset.idx_to_class[label_idx]
        
        with torch.no_grad():
            mu, _ = model.encoder(original_img)
            z = mu # [1, 64]
            
            img_pred = torch.randn_like(original_img)
            
            for t in tqdm(range(NUM_TIMESTEPS - 1, -1, -1), desc=f"Img {i+1} ({class_name})", leave=False):
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = model.decoder(img_pred, t_tensor, z)
                img_pred = noise_scheduler.step(noise_pred, t, img_pred).prev_sample

        axes[0, i].imshow(unnormalize(original_img[0]))
        axes[0, i].set_title(f"Orig: {class_name}", fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(unnormalize(img_pred[0]))
        axes[1, i].set_title("Rec Semantic DGAE", fontsize=10)
        axes[1, i].axis('off')

    plt.suptitle(f"Reconstrução - {title}", fontsize=16)
    plt.savefig(f"{out_prefix}_reconstruct.png")
    print(f"Salvo: {out_prefix}_reconstruct.png")

# ==============================================================================
# 4. CLUSTERIZAÇÃO
# ==============================================================================
def analisar_latent_space(model, dataset):
    print("\n--- Analisando Latent Space (Vector 64d) ---")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extraindo Features"):
            imgs = batch[0].to(device)
            labels = batch[1]
            mu, _ = model.encoder(imgs) 
            
            # Já é vetor [B, 64], apenas converte para numpy
            embeddings_list.append(mu.cpu().numpy())
            labels_list.extend(labels.numpy())

    X = np.vstack(embeddings_list)
    y_true = np.array(labels_list)
    
    unique_indices = np.unique(y_true)
    n_classes = len(unique_indices)
    n_clusters = max(n_classes, 2)
    class_names = [dataset.idx_to_class[i] for i in unique_indices]

    print(f"Features: {X.shape} (Latente=64) | Classes: {class_names}")

    resultados = {}

    def calc_metrics(X_transformed, y_true, y_pred):
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        sil = silhouette_score(X_transformed, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        return {'ari': ari, 'nmi': nmi, 'sil': sil, 'viz': X_transformed}

    # Reduções
    pca = PCA(n_components=min(10, X.shape[1])) 
    X_pca = pca.fit_transform(X)
    y_pred_pca = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_pca)
    resultados['PCA'] = calc_metrics(X_pca[:, :2], y_true, y_pred_pca)

    tsne = TSNE(n_components=2, perplexity=min(30, len(X)-1), random_state=42)
    X_tsne = tsne.fit_transform(X)
    y_pred_tsne = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_tsne)
    resultados['t-SNE'] = calc_metrics(X_tsne, y_true, y_pred_tsne)

    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    y_pred_umap = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_umap)
    resultados['UMAP'] = calc_metrics(X_umap, y_true, y_pred_umap)

    # --- PLOT ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7)) 
    metodos = ['PCA', 't-SNE', 'UMAP']
    cmap = plt.get_cmap('tab20', len(unique_indices)) if len(unique_indices) > 10 else plt.get_cmap('tab10', len(unique_indices))

    for i, m in enumerate(metodos):
        ax = axes[i]
        res = resultados[m]
        
        scatter = ax.scatter(res['viz'][:, 0], res['viz'][:, 1], c=y_true, cmap=cmap, s=20, alpha=0.7)
        ax.set_title(f"{m}", fontsize=14, fontweight='bold')
        
        stats = f"ARI: {res['ari']:.3f}\nNMI: {res['nmi']:.3f}\nSil: {res['sil']:.3f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, verticalalignment='top', bbox=props, fontsize=11)

    # Legenda
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=name,
               markerfacecolor=cmap(i/(len(unique_indices)-1)) if len(unique_indices) > 1 else cmap(0), 
               markersize=10)
        for i, name in enumerate(class_names)
    ]
    
    fig.legend(handles=legend_elements, loc='lower right', title="Classes", fontsize=12, title_fontsize=14)
    plt.subplots_adjust(right=0.88, wspace=0.2) 

    plt.suptitle(f"Clusterização Semantic/Vector - {title}", fontsize=18)
    save_path = f"{out_prefix}_clustering.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Salvo: {save_path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
        # exit() 

    print(f"Inicializando Avaliação para: {MODEL_TYPE}")
    dataset = CorelDataset(DATA_DIR, IMAGE_SIZE)
    
    # Inicializa arquitetura Semantic (Vector 64)
    model = DGAE_Semantic(LATENT_DIM).to(device)

    print(f"Carregando pesos: {MODEL_PATH}")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("✓ Pesos carregados com sucesso.")
    except Exception as e:
        print(f"⚠️ Erro ao carregar pesos: {e}")
        print("Verifique se está usando os pesos corretos do treino Vector VAE (não os do Spatial).")

    gerar_reconstrucoes(model, dataset)
    analisar_latent_space(model, dataset)