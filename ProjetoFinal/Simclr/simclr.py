import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

Augmented = True   

if Augmented:
    DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel augmented"
else:
    DATA_DIR = "/home/rick/Desktop/MO433/Projeto Final/corel"

# --- CONFIGURAÇÕES MELHORADAS ---
CONFIG = {
    "data_dir": DATA_DIR,
    "save_dir": ".",
    # Tente aumentar para 32, 64 ou 128. Batch Size 8 QUEBRA o SimCLR.
    "batch_size": 64,  
    "image_size": 128, # Reduzi para 128 para permitir batch maior na GPU
    "epochs": 200,
    "lr": 1e-4, # Learning rate menor para ResNet
    "temperature": 0.07, # 0.5 é muito alto, 0.07 é o padrão do paper
    "latent_dim": 128,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# --- 1. AUGMENTATIONS DO PAPER (CRUCIAIS) ---
# SimCLR precisa de ColorJitter forte e Blur para não aprender atalhos de cor
def get_simclr_transforms(size):
    color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    return transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)), # Crop agressivo
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8), # Distorção de cor é vital
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# --- 2. MODELO COM TRANSFER LEARNING ---
class SimCLR_ResNet(nn.Module):
    def __init__(self, latent_dim=128, pretrained=True):
        super().__init__()
        
        # AQUI ESTÁ O SEGREDO: Carregar pesos da ImageNet
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        print(f"Carregando ResNet18 {'com pesos ImageNet' if pretrained else 'do zero'}...")
        
        self.encoder = models.resnet18(weights=weights)
        
        # Opcional: Congelar as primeiras camadas para não destruir os filtros básicos
        # Descomente abaixo se o dataset for MUITO pequeno (<500 imagens)
        # for name, param in self.encoder.named_parameters():
        #     if "layer4" not in name and "fc" not in name: # Treina só layer4 pra frente
        #         param.requires_grad = False

        self.feature_dim = self.encoder.fc.in_features
        
        # Remove a FC original (classificação) e coloca Identidade
        self.encoder.fc = nn.Identity()

        # Projection Head (Treinável)
        self.projector = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):
        h = self.encoder(x)       # Features Ricas (Use ISSO para clusterização)
        z = self.projector(h)     # Projeção (Use ISSO apenas para calcular a Loss)
        return h, z

# --- 3. NT-Xent LOSS ---
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        # Implementação otimizada para estabilidade
        batch_size = z_i.shape[0] # Pega dinâmico para evitar erro no último batch
        N = 2 * batch_size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # Máscara para remover auto-similaridade
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        # Precisamos construir os logits positivos e negativos corretamente
        # Essa é uma versão simplificada:
        # Para cada imagem i, o positivo é (i + batch_size). Todos os outros são negativos.
        
        labels = torch.arange(batch_size).to(self.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Removemos a diagonal principal (self-similarity) da matriz de similaridade N x N
        mask = torch.eye(N, dtype=bool).to(self.device)
        sim.masked_fill_(mask, -9e15) # -Infinito
        
        loss = F.cross_entropy(sim, labels)
        return loss / N # Média por amostra

# --- DATASET ---
class SimpleFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.files = sorted(glob.glob(os.path.join(root, "*.png")) + glob.glob(os.path.join(root, "*.jpg")))
        self.transform = transform
        if len(self.files) == 0: raise RuntimeError("Sem imagens encontradas")
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        try:
            img = Image.open(self.files[i]).convert("RGB")
            return self.transform(img), self.transform(img)
        except:
            return self.__getitem__((i + 1) % len(self))
        
# --- TREINO ---
def train():
    if not os.path.exists(CONFIG['data_dir']):
        print(f"Erro: Pasta {CONFIG['data_dir']} não existe.")
        return

    dataset = SimpleFolderDataset(CONFIG['data_dir'], get_simclr_transforms(CONFIG['image_size']))
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True, num_workers=4)
    
    # Pretrained=True é a chave para funcionar em dataset pequeno
    model = SimCLR_ResNet(latent_dim=CONFIG['latent_dim'], pretrained=True).to(CONFIG['device'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'], weight_decay=1e-5)
    
    # Loss precisa ser instanciada dentro do loop ou ajustada dinamicamente, 
    # mas aqui instanciamos a classe auxiliar
    loss_fn = NTXentLoss(CONFIG['batch_size'], CONFIG['temperature'], CONFIG['device'])
    
    print(f"Iniciando Fine-Tuning SimCLR | Batch: {CONFIG['batch_size']}")
    
    loss_history = []
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0
        
        for (x_i, x_j) in loader:
            x_i, x_j = x_i.to(CONFIG['device']), x_j.to(CONFIG['device'])
            
            optimizer.zero_grad()
            
            h_i, z_i = model(x_i)
            h_j, z_j = model(x_j)
            
            loss = loss_fn(z_i, z_j)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{CONFIG['epochs']} - Loss: {avg_loss:.4f}")
        
    # Salvar
    torch.save(model.state_dict(), "simclr_aug_model.pth")
    
    plt.plot(loss_history)
    plt.title("SimCLR Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig("simclr_loss.png")
    print("Treino salvo!")

if __name__ == "__main__":
    train()