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
from matplotlib.lines import Line2D # Importado para criar a legenda customizada
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
MODEL_TYPE = "Spatial"  # Opções: "Spatial", "Vector"
Augmented = False        # Se True, usa caminhos da pasta augmented

# Definição de Caminhos
if MODEL_TYPE == "Spatial":
    if Augmented:
        MODEL_PATH = "dgae_spatial_aug.pth"
        DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel augmented"
        out_prefix = "spatial_aug"
        title = "DGAE Spatial 16x16 (Augmented)"
    else:
        MODEL_PATH = "dgae_spatial_base.pth"
        DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel"
        out_prefix = "spatial_base"
        title = "DGAE Spatial 16x16 (Base)"
    
    LATENT_CHANNELS = 4  
    IMAGE_SIZE = 128
    BATCH_SIZE = 32
    NUM_TIMESTEPS = 1000 
    NUM_RECONSTRUCTIONS = 4 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. ARQUITETURAS SPATIAL
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

class Spatial_VAE_Encoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3 
        )
        self.conv_mu = nn.Conv2d(256, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(256, latent_channels, kernel_size=1)

    def forward(self, x):
        feat = self.features(x)
        mu = self.conv_mu(feat)
        log_var = self.conv_logvar(feat)
        return mu, log_var

class Spatial_Diffusion_Decoder(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(128), nn.Linear(128, 128), nn.GELU())
        in_channels = 3 + latent_channels
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU()) 
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(8, 256), nn.GELU()) 
        self.bottleneck = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.GELU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU()) 
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU())  
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x_noisy, t, z_spatial):
        z_upsampled = F.interpolate(z_spatial, size=x_noisy.shape[2:], mode='nearest')
        x_input = torch.cat([x_noisy, z_upsampled], dim=1)
        t_emb = self.time_mlp(t)[..., None, None]
        x1 = self.down1(x_input)
        x2 = self.down2(x1) + t_emb 
        x3 = self.down3(x2)
        mid = self.bottleneck(x3)
        return self.out(self.up2(self.up1(mid + x3) + x2) + x1)

class DGAE_Spatial(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        self.encoder = Spatial_VAE_Encoder(latent_channels)
        self.decoder = Spatial_Diffusion_Decoder(latent_channels)

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
# 3. RECONSTRUÇÃO (Com Rótulos)
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
            z = mu 
            img_pred = torch.randn_like(original_img)
            
            for t in tqdm(range(NUM_TIMESTEPS - 1, -1, -1), desc=f"Img {i+1} ({class_name})", leave=False):
                t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
                noise_pred = model.decoder(img_pred, t_tensor, z)
                img_pred = noise_scheduler.step(noise_pred, t, img_pred).prev_sample

        axes[0, i].imshow(unnormalize(original_img[0]))
        axes[0, i].set_title(f"Orig: {class_name}", fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(unnormalize(img_pred[0]))
        axes[1, i].set_title("Rec DGAE Spatial", fontsize=10)
        axes[1, i].axis('off')

    plt.suptitle(f"Reconstrução - {title}", fontsize=16)
    plt.savefig(f"{out_prefix}_reconstruct.png")
    print(f"Salvo: {out_prefix}_reconstruct.png")

# ==============================================================================
# 4. CLUSTERIZAÇÃO (Com Rótulos na Legenda)
# ==============================================================================
def analisar_latent_space(model, dataset):
    print("\n--- Analisando Latent Space ---")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extraindo Features"):
            imgs = batch[0].to(device)
            labels = batch[1]
            mu, _ = model.encoder(imgs) 
            flat_mu = mu.view(mu.size(0), -1).cpu().numpy()
            embeddings_list.append(flat_mu)
            labels_list.extend(labels.numpy())

    X = np.vstack(embeddings_list)
    y_true = np.array(labels_list)
    
    # Identifica classes presentes neste teste
    unique_indices = np.unique(y_true)
    n_classes = len(unique_indices)
    n_clusters = max(n_classes, 2)
    
    # Recupera nomes reais das classes
    class_names = [dataset.idx_to_class[i] for i in unique_indices]

    print(f"Features: {X.shape} | Classes: {class_names}")

    resultados = {}

    def calc_metrics(X_transformed, y_true, y_pred):
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        sil = silhouette_score(X_transformed, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        return {'ari': ari, 'nmi': nmi, 'sil': sil, 'viz': X_transformed}

    # Reduções de Dimensionalidade
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

    # --- PLOT COM LEGENDA DE CLASSES ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 7)) 
    metodos = ['PCA', 't-SNE', 'UMAP']
    
    # Define Colormap Discreto
    cmap = plt.get_cmap('tab20', len(unique_indices)) if len(unique_indices) > 10 else plt.get_cmap('tab10', len(unique_indices))

    for i, m in enumerate(metodos):
        ax = axes[i]
        res = resultados[m]
        
        # Mapeia os índices das classes (y_true) para cores consistentes
        # Precisamos garantir que o índice '0' use a cor 0 do cmap, etc.
        # Como unique_indices pode ser [0, 5, 8], usamos um map auxiliar se necessário, 
        # mas aqui faremos mapeamento direto para simplificar o scatter
        
        scatter = ax.scatter(res['viz'][:, 0], res['viz'][:, 1], c=y_true, cmap=cmap, s=20, alpha=0.7)
        ax.set_title(f"{m}", fontsize=14, fontweight='bold')
        
        # Caixa de Texto com Métricas
        stats = f"ARI: {res['ari']:.3f}\nNMI: {res['nmi']:.3f}\nSil: {res['sil']:.3f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.9)
        ax.text(0.05, 0.95, stats, transform=ax.transAxes, verticalalignment='top', bbox=props, fontsize=11)

    # --- CRIAÇÃO DA LEGENDA CUSTOMIZADA ---
    # Cria "handles" falsos para a legenda baseados nas cores do scatter
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=name,
               markerfacecolor=cmap(i/(len(unique_indices)-1)) if len(unique_indices) > 1 else cmap(0), 
               markersize=10)
        for i, name in enumerate(class_names)
    ]
    
    # Adiciona a legenda ao lado direito da figura inteira
    fig.legend(handles=legend_elements, loc='lower right', title="Classes", fontsize=12, title_fontsize=14)
    plt.subplots_adjust(right=0.88, wspace=0.2) # Ajusta margem para caber a legenda

    plt.suptitle(f"Clusterização Spatial - {title}", fontsize=18)
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
    model = DGAE_Spatial(LATENT_CHANNELS).to(device)

    print(f"Carregando pesos: {MODEL_PATH}")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        print("✓ Pesos carregados com sucesso.")
    except Exception as e:
        print(f"⚠️ Erro ao carregar pesos: {e}")

    gerar_reconstrucoes(model, dataset)
    analisar_latent_space(model, dataset)