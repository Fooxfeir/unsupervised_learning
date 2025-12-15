import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from diffusers import DDPMScheduler
import glob
import os
from PIL import Image
import math
from tqdm.auto import tqdm
import lpips
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURAÇÕES ---
# Ajuste conforme seu ambiente
Augmented = True
if Augmented:
    DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel augmented"
    output_path = "dgae_semantic_aug.pth"
else:
    DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel"
    output_path = "dgae_semantic_base.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 32         # Batches maiores ajudam na estabilidade da KL
LATENT_DIM = 64         # Vetor 1D
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
NUM_TIMESTEPS = 1000

# Pesos da Loss (Balanceamento delicado)
ALPHA_DSM = 1.0
BETA_KL = 1e-6          # Muito baixo para não destruir a separação dos clusters
ETA_LPIPS = 0.1         # Ajuda na textura

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. ENCODER VECTORIAL VAE (Global Pooling + Probabilístico) ---
class Global_VAE_Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        # ResNet18 Pré-treinada
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Removemos FC e AvgPool originais
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

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

# --- 2. DECODER DE DIFUSÃO (Vetorial) ---
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

class Vector_Diffusion_Decoder(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(128), nn.Linear(128, 128), nn.GELU())
        
        # Input: 3 canais imagem + latent_dim (expandido)
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
        # Expande para [B, 64, H, W]
        z_expanded = z_spatial.expand(-1, -1, x.shape[2], x.shape[3])
        
        x_input = torch.cat([x, z_expanded], dim=1)
        
        t_emb = self.time_mlp(t)[..., None, None]
        x1 = self.down1(x_input)
        x2 = self.down2(x1) + t_emb 
        x3 = self.down3(x2)
        mid = self.bottleneck(x3)
        return self.out(self.up2(self.up1(mid + x3) + x2) + x1)

class DGAE_Vector_VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = Global_VAE_Encoder(latent_dim)
        self.decoder = Vector_Diffusion_Decoder(latent_dim)

    def forward(self, x, t, noise):
        # 1. Encode Probabilístico
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mu, log_var)
        
        # 2. Decode
        noise_pred = self.decoder(noise, t, z)
        return noise_pred, mu, log_var

# --- 3. AUXILIAR: PREDIÇÃO DE X0 (Para LPIPS) ---
def predict_x0(noisy_sample, noise_pred, timesteps, scheduler):
    alphas_cumprod = scheduler.alphas_cumprod.to(noisy_sample.device)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (noisy_sample - (beta_prod_t ** 0.5) * noise_pred) / (alpha_prod_t ** 0.5)
    return torch.clamp(pred_original_sample, -1, 1)

# --- 4. DATASET E TREINO ---
class CorelDataset(Dataset):
    def __init__(self, data_dir, image_size):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.png")) + glob.glob(os.path.join(data_dir, "*.jpg")))
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx): return self.transform(Image.open(self.files[idx]).convert('RGB'))

if __name__ == "__main__":
    print(f"Iniciando DGAE Vector VAE (Cluster + Qualidade)... GPU: {torch.cuda.get_device_name(0)}")
    
    # LPIPS
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    loss_fn_lpips.requires_grad_(False)

    dataset = CorelDataset(DATA_DIR, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    model = DGAE_Vector_VAE(LATENT_DIM).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        
        for clean_images in pbar:
            clean_images = clean_images.to(device)
            bs = clean_images.shape[0]
            
            # Ruído
            noise = torch.randn_like(clean_images)
            timesteps = torch.randint(0, NUM_TIMESTEPS, (bs,), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # Forward
            noise_pred, mu, log_var = model(clean_images, timesteps, noisy_images)
            
            # --- CALCULO DAS LOSSES ---
            
            # 1. MSE (Estrutura)
            loss_mse = F.mse_loss(noise_pred, noise)
            
            # 2. KL (Regularidade do Latente)
            # Para vetores 1D, somamos apenas na dim=1
            loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / bs
            
            # 3. LPIPS (Textura)
            pred_x0 = predict_x0(noisy_images, noise_pred, timesteps, noise_scheduler)
            loss_lpips_val = loss_fn_lpips(pred_x0, clean_images).mean()
            
            # Soma Ponderada
            loss = (ALPHA_DSM * loss_mse) + (BETA_KL * loss_kl) + (ETA_LPIPS * loss_lpips_val)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({
                "MSE": f"{loss_mse.item():.4f}", 
                "KL": f"{loss_kl.item():.4f}",
                "LPIPS": f"{loss_lpips_val.item():.4f}"
            })
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), output_path)
    print(f"\n✓ Modelo Final Salvo: {output_path}")