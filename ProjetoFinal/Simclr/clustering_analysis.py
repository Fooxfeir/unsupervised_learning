import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 
from tqdm.auto import tqdm
import numpy as np
import warnings

# Imports para Análise de Clusterização
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap 
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

warnings.filterwarnings("ignore")

# ==============================================================================
# --- CONFIGURAÇÕES DE AVALIAÇÃO ---
# ==============================================================================
Augmented = True   # <--- ALTERE AQUI: True para Augmented, False para Base

# Definição Automática de Caminhos e Títulos
if Augmented:
    # Ajuste o nome do arquivo .pth do seu modelo treinado com Augmentation
    MODEL_PATH = "simclr_aug_model.pth" 
    DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel augmented"
    out_prefix = "simclr_aug"
    title = "SimCLR (Augmented)"
else:
    # Ajuste o nome do arquivo .pth do seu modelo treinado na Base
    MODEL_PATH = "simclr_base_model.pth"
    DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel"
    out_prefix = "simclr_base"
    title = "SimCLR (Base)"

IMAGE_SIZE = 128
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 1. ARQUITETURA SIMCLR (Encoder/Backbone)
# ==============================================================================
class SimCLR_ResNet(nn.Module):
    def __init__(self, base_model="resnet18", out_dim=128):
        super(SimCLR_ResNet, self).__init__()
        
        # Carrega ResNet padrão
        self.backbone = models.resnet18(weights=None)
        dim_mlp = self.backbone.fc.in_features
        
        # Remove a última camada FC original
        # Substituímos por Identidade para pegar as features puras (h)
        self.backbone.fc = nn.Identity()
        
        # Projection Head (z) - Geralmente ignorada no eval de clustering
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, out_dim)
        )

    def forward(self, x):
        # Retorna a representação h (antes do projector) para clustering
        h = self.backbone(x)
        return h

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
                # Extrai classe do nome (ex: "1_100.jpg" -> "1")
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

# ==============================================================================
# 3. FUNÇÃO DE CLUSTERIZAÇÃO
# ==============================================================================
def analisar_latent_space(model, dataset):
    print(f"\n--- Analisando Representações: {title} ---")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    embeddings_list = []
    labels_list = []

    model.eval()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extraindo Features"):
            imgs = imgs.to(device)
            
            # No SimCLR, pegamos a saída do backbone (features h)
            features = model(imgs) 
            
            # Flatten se necessário (ResNet já sai [B, 512] após Identity)
            features = features.view(features.size(0), -1).cpu().numpy()
            
            embeddings_list.append(features)
            labels_list.extend(labels.numpy())

    X = np.vstack(embeddings_list)
    y_true = np.array(labels_list)
    
    unique_indices = np.unique(y_true)
    n_classes = len(unique_indices)
    n_clusters = max(n_classes, 2)
    class_names = [dataset.idx_to_class[i] for i in unique_indices]

    print(f"Features Extraídas: {X.shape} | Classes: {class_names}")

    resultados = {}

    def calc_metrics(X_transformed, y_true, y_pred):
        ari = adjusted_rand_score(y_true, y_pred)
        nmi = normalized_mutual_info_score(y_true, y_pred)
        sil = silhouette_score(X_transformed, y_pred) if len(np.unique(y_pred)) > 1 else 0.0
        return {'ari': ari, 'nmi': nmi, 'sil': sil, 'viz': X_transformed}

    # --- 1. PCA ---
    print("Calculando PCA...")
    pca = PCA(n_components=min(10, X.shape[1])) 
    X_pca = pca.fit_transform(X)
    y_pred_pca = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_pca)
    resultados['PCA'] = calc_metrics(X_pca[:, :2], y_true, y_pred_pca)

    # --- 2. t-SNE ---
    print("Calculando t-SNE...")
    perp = min(30, len(X)-1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42)
    X_tsne = tsne.fit_transform(X)
    y_pred_tsne = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_tsne)
    resultados['t-SNE'] = calc_metrics(X_tsne, y_true, y_pred_tsne)

    # --- 3. UMAP ---
    print("Calculando UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)
    y_pred_umap = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_umap)
    resultados['UMAP'] = calc_metrics(X_umap, y_true, y_pred_umap)

    # --- PLOTAGEM ---
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

    # --- LEGENDA ---
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=name,
               markerfacecolor=cmap(i/(len(unique_indices)-1)) if len(unique_indices) > 1 else cmap(0), 
               markersize=10)
        for i, name in enumerate(class_names)
    ]
    
    fig.legend(handles=legend_elements, loc='lower right', title="Classes", fontsize=12, title_fontsize=14)
    plt.subplots_adjust(right=0.88, wspace=0.2) 

    plt.suptitle(f"Clusterização Contrastiva - {title}", fontsize=18)
    
    save_path = f"{out_prefix}_clustering.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Gráfico salvo em: {save_path}")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo SimCLR não encontrado em {MODEL_PATH}")
        print("Verifique se o nome do arquivo .pth está correto na seção de Configurações.")
        # exit() 

    print(f"Inicializando Avaliação SimCLR ({'AUGMENTED' if Augmented else 'BASE'})...")
    dataset = CorelDataset(DATA_DIR, IMAGE_SIZE)
    
    model = SimCLR_ResNet().to(device)

    print(f"Carregando pesos: {MODEL_PATH}")
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                new_state_dict[k] = v 
            elif not k.startswith("projector."): 
                # Se salvou o modelo inteiro sem prefixo backbone, adiciona
                new_state_dict[f"backbone.{k}"] = v
                
        # Tenta carregar ignorando erros de chaves extras (projector) ou faltantes
        model.load_state_dict(state_dict, strict=False)
        print("✓ Pesos carregados (Strict=False).")
            
    except Exception as e:
        print(f"⚠️ Erro ao carregar pesos: {e}")

    analisar_latent_space(model, dataset)