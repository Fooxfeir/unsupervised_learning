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
Augmented = True
if Augmented:
    DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel augmented"
    output_path = "dgae_spatial_aug.pth"
else:
    DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel"
    output_path = "dgae_spatial_base.pth"

IMAGE_SIZE = 128
BATCH_SIZE = 32
# No paper, usam f8 ou f16. Vamos usar f32 (padrão ResNet) ou cortar camadas.
# ResNet18 completa reduz 32x. 128/32 = 4x4. É muito pequeno.
# Vamos cortar a layer4 para ficar com f16 (128/16 = 8x8).
LATENT_CHANNELS = 4     # O paper sugere canais baixos (ex: 4 ou 8) para compressão [cite: 284]
NUM_EPOCHS = 200
LEARNING_RATE = 1e-4
NUM_TIMESTEPS = 1000

# Pesos da Loss (O paper usa alpha, beta, eta) [cite: 169]
ALPHA_DSM = 1.0
BETA_KL = 1e-6          
ETA_LPIPS = 0.1         

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. SPATIAL VAE ENCODER (Preserva HxW) ---
class Spatial_VAE_Encoder(nn.Module):
    def __init__(self, latent_channels=4):
        super().__init__()
        # ResNet18 Pré-treinada
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Usamos até a layer3 para ter downsampling de 16x (128 -> 8x8)
        # Se usarmos layer4, seria 32x (128 -> 4x4), o que pode perder muita info espacial.
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3 
            # resnet.layer4 (Removido para manter f=16)
        )
        
        # A saída da layer3 tem 256 canais
        self.conv_mu = nn.Conv2d(256, latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv2d(256, latent_channels, kernel_size=1)

    def forward(self, x):
        # x: [B, 3, 128, 128]
        feat = self.features(x) # [B, 256, 8, 8] (para f=16)
        
        mu = self.conv_mu(feat)        # [B, 4, 8, 8]
        log_var = self.conv_logvar(feat) # [B, 4, 8, 8]
        
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

# --- 2. DECODER DE DIFUSÃO (Condicionamento Espacial) ---
# O Paper especifica: Upsample latent -> Concatenate with noise 

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

class Spatial_Diffusion_Decoder(nn.Module):
    def __init__(self, latent_channels):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPositionEmbeddings(128), nn.Linear(128, 128), nn.GELU())
        
        # O input será: 3 canais (ruído) + latent_channels (upsampled z)
        in_channels = 3 + latent_channels
        
        # Arquitetura U-Net simples
        self.down1 = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU()) 
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.GroupNorm(8, 256), nn.GELU()) 
        self.bottleneck = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.GELU())
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.GroupNorm(8, 128), nn.GELU()) 
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.GroupNorm(8, 64), nn.GELU())  
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x_noisy, t, z_spatial):
        # 1. Upsample do Latente para o tamanho da imagem (Nearest Neighbor) 
        # z_spatial: [B, 4, 8, 8] -> [B, 4, 128, 128]
        z_upsampled = F.interpolate(z_spatial, size=x_noisy.shape[2:], mode='nearest')
        
        # 2. Concatenação 
        x_input = torch.cat([x_noisy, z_upsampled], dim=1)
        
        t_emb = self.time_mlp(t)[..., None, None]
        
        # Forward U-Net
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

    def forward(self, x, t, noise):
        # 1. Encode Espacial
        mu, log_var = self.encoder(x)
        z = self.encoder.reparameterize(mu, log_var)
        
        # 2. Decode (z é passado como mapa espacial)
        noise_pred = self.decoder(noise, t, z)
        return noise_pred, mu, log_var

# --- FUNÇÃO AUXILIAR ---
def predict_x0(noisy_sample, noise_pred, timesteps, scheduler):
    alphas_cumprod = scheduler.alphas_cumprod.to(noisy_sample.device)
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (noisy_sample - (beta_prod_t ** 0.5) * noise_pred) / (alpha_prod_t ** 0.5)
    return torch.clamp(pred_original_sample, -1, 1)

# --- DATASET E TREINO ---
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
    print(f"Iniciando DGAE SPATIAL (Fiel ao Paper)... GPU: {torch.cuda.get_device_name(0)}")
    
    loss_fn_lpips = lpips.LPIPS(net='vgg').to(device)
    loss_fn_lpips.requires_grad_(False)

    dataset = CorelDataset(DATA_DIR, IMAGE_SIZE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    noise_scheduler = DDPMScheduler(num_train_timesteps=NUM_TIMESTEPS)
    model = DGAE_Spatial(LATENT_CHANNELS).to(device)
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
            
            # 1. DSM (MSE no ruído) [cite: 144, 155]
            loss_mse = F.mse_loss(noise_pred, noise)
            
            # 2. KL Divergence (Soma sobre canais e dimensões espaciais) [cite: 96]
            # O tensor é [B, C, H, W]. Somamos tudo exceto Batch.
            loss_kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / bs
            
            # 3. LPIPS no x0 predito [cite: 167]
            pred_x0 = predict_x0(noisy_images, noise_pred, timesteps, noise_scheduler)
            loss_lpips_val = loss_fn_lpips(pred_x0, clean_images).mean()
            
            # Soma Ponderada [cite: 169]
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
    print(f"\n✓ Modelo Spatial Final Salvo: {output_path}")