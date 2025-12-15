import torch
from accelerate import utils
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import math
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import torch.nn.functional as F
from diffusers.utils import convert_state_dict_to_diffusers
import gc
import os
import glob
from PIL import Image
from datetime import datetime

# --- CONFIGURAÇÃO GLOBAL ---
formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

# SEU DICIONÁRIO (Mantemos ele, o código vai pegar só o prompt da classe escolhida)
CLASS_PROMPT_DICT = {
    "1": "a photorealistic photo of the british royal guard",
    "2": "a photorealistic photo of a steam locomotive",
    "3": "a photorealistic photo of a pie",
    "4": "a photorealistic photo of vegetables",
    "5": "a photorealistic photo of a snowy landscape",
    "6": "a photorealistic photo of a orange sunset",
}

# --- CLASSE DE DATASET COM FILTRO ---
class SingleClassCorelDataset(Dataset):
    def __init__(self, data_dir, tokenizer, target_class_id, prompt_dict, image_size=512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.target_class_id = str(target_class_id) # Ex: "1"
        self.prompt_dict = prompt_dict
        self.image_size = image_size
        
        # 1. Listar todos os arquivos
        all_files = glob.glob(os.path.join(data_dir, "*.png"))
        if len(all_files) == 0:
            all_files = glob.glob(os.path.join(data_dir, "*.jpg"))
            
        # 2. FILTRAR: Manter apenas imagens da classe alvo
        self.image_paths = []
        for file_path in all_files:
            filename = os.path.basename(file_path)
            # Verifica se o arquivo começa com "1_" (por exemplo)
            if filename.startswith(f"{self.target_class_id}_"):
                self.image_paths.append(file_path)
        
        self.image_paths.sort()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"Nenhuma imagem encontrada para a classe '{self.target_class_id}' em {data_dir}. Verifique o nome dos arquivos (ex: 1_xxx.png)")

        print(f"✓ Dataset filtrado para classe {self.target_class_id}: {len(self.image_paths)} imagens encontradas.")

        self.transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size), # Center crop é melhor para objetos específicos
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]) 
        ])

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        # Pega o prompt fixo da classe alvo
        if self.target_class_id in self.prompt_dict:
            caption = self.prompt_dict[self.target_class_id]
        else:
            caption = f"a photo of class {self.target_class_id}"

        input_ids = self.tokenizer(
            caption, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids[0]

        pixel_values = self.transforms(image)

        return {"pixel_values": pixel_values, "input_ids": input_ids}

def main():
    # --- ESCOLHA AQUI A CLASSE QUE VOCÊ QUER TREINAR ---
    TARGET_CLASS_ID = "0001"  # <--- Mude para "1", "2", "3"...
    # ---------------------------------------------------

    utils.write_basic_config()
    
    # Configurações
    output_dir                      = f"corel_class_{TARGET_CLASS_ID}_output" # Pasta específica
    pretrained_model_name_or_path   = "runwayml/stable-diffusion-v1-5"
    train_data_dir                  = "./corel"
    
    # Hiperparâmetros RTX 2060
    train_batch_size                = 1
    gradient_accumulation_steps     = 4
    resolution                      = 512
    learning_rate                   = 1e-4
    
    # Lógica de Épocas para Dataset Pequeno (Single Class)
    # Vamos supor que cada classe tem ~60 imagens.
    # 60 imgs / batch 1 = 60 passos por época.
    # Para chegar em 1500 passos totais -> 1500 / 60 = 25 épocas.
    num_train_epochs                = 30  # Aumentamos pq agora são poucas imagens
    
    # LoRA Config
    lora_rank                       = 4
    lora_alpha                      = 4
    train_text_encoder              = True # False é mais rápido e gasta menos VRAM

    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps,
        mixed_precision             = "fp16"
    )
    device = accelerator.device

    # Load Models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, torch_dtype=torch.float16
    ).to(device)
    tokenizer, text_encoder, vae, unet = pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet

    # Otimizações
    unet.enable_gradient_checkpointing()
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    vae.to('cpu')
    torch.cuda.empty_cache()

    # LoRA UNet
    unet_lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet.add_adapter(unet_lora_config)
    
    for param in unet.parameters():
        if param.requires_grad: param.data = param.to(torch.float32)

    # --- CARREGA DATASET FILTRADO ---
    print(f"\nPreparando dataset APENAS para a classe {TARGET_CLASS_ID}...")
    train_dataset = SingleClassCorelDataset(
        data_dir=train_data_dir,
        tokenizer=tokenizer,
        target_class_id=TARGET_CLASS_ID, # Passamos o ID alvo
        prompt_dict=CLASS_PROMPT_DICT,
        image_size=resolution
    )
    
    if len(train_dataset) < 10:
        print("⚠ AVISO: Poucas imagens encontradas. O overfitting será rápido.")

    def collate_fn(examples):
        pixel_values = torch.stack([ex["pixel_values"] for ex in examples]).to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([ex["input_ids"] for ex in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, 
        batch_size=train_batch_size, num_workers=6
    )

    # Otimizador
    params_to_optimize = list(filter(lambda p: p.requires_grad, unet.parameters()))
    optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=1e-2)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer)

    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # --- LOOP DE TREINO ---
    max_train_steps = num_train_epochs * len(train_dataloader)
    print(f"\nIniciando Treino: {num_train_epochs} épocas, Total Passos: {max_train_steps}")
    print(f"Tempo estimado na 2060: ~{(max_train_steps/1.2)/60:.1f} minutos\n")
    
    progress_bar = tqdm(range(max_train_steps), desc="Steps")

    for epoch in range(num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # VAE Encode
                vae.to(device)
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float16)).latent_dist.sample() * 0.18215
                vae.to('cpu')

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                progress_bar.set_postfix({"loss": f"{loss.detach().item():.4f}"})

    # --- SALVAR ---
    print("\nSalvando modelo...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(unet)))
        
        # Nome específico para a classe
        output_name = f"lora_class_{TARGET_CLASS_ID}_{formatted_date}.safetensors"
        
        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_layers,
            safe_serialization=True,
            weight_name=output_name
        )
        print(f"✓ Modelo salvo em: {output_dir}/{output_name}")

if __name__ == "__main__":
    main()