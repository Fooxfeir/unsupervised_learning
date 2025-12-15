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
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- CONFIGURAÇÃO GLOBAL ---
formatted_date = datetime.now().strftime(r'%Y%m%d-%H%M%S')

# SEU DICIONÁRIO DE PROMPTS
CLASS_PROMPT_DICT = {
    "1": "a photorealistic photo of the british royal guard",
    "2": "a photorealistic photo of a steam locomotive",
    "3": "a photorealistic photo of a pie",
    "4": "a photorealistic photo of different vegetables",
    "5": "a photorealistic photo of a snowy landscape",
    "6": "a photorealistic photo of a orange sunset",
}

# --- CLASSE DE DATASET ---
class CorelSDDataset(Dataset):
    def __init__(self, data_dir, tokenizer, prompt_dict=None, image_size=512, center_crop=True, random_flip=True):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.prompt_dict = prompt_dict if prompt_dict is not None else {}
        self.image_size = image_size
        
        # Procura imagens
        self.image_paths = glob.glob(os.path.join(data_dir, "*.png"))
        self.image_paths.sort()
        if len(self.image_paths) == 0:
            self.image_paths = glob.glob(os.path.join(data_dir, "*.jpg"))
            
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {data_dir}")

        # Transformações para o padrão do Stable Diffusion (-1 a 1)
        self.transforms = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
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

        # Extrai classe do nome (Ex: 1_100.png -> "1")
        filename = os.path.basename(image_path)
        try:
            class_num = int(filename.split('_')[0])
            class_key = str(class_num) 
        except (ValueError, IndexError):
            class_key = "unknown"

        # Define o prompt usando o dicionário
        if class_key in self.prompt_dict:
            caption = self.prompt_dict[class_key]
        else:
            caption = f"a photorealistic photo of class {class_key}"

        # Tokeniza o texto
        input_ids = self.tokenizer(
            caption, 
            max_length=self.tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        ).input_ids[0]

        pixel_values = self.transforms(image)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids
        }

def main():
    # Verifica GPU
    if torch.cuda.is_available():
        print(f"✓ GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠ WARNING: No GPU detected. Training will be extremely slow.")

    utils.write_basic_config()

    # --- HIPERPARÂMETROS ---
    output_dir                      = "corel_lora_output"
    pretrained_model_name_or_path   = "runwayml/stable-diffusion-v1-5"
    train_data_dir                  = "./corel"  # Pasta onde estão as imagens
    
    # LoRA params
    lora_rank                       = 4
    lora_alpha                      = 4
    text_encoder_lora_rank          = 4     
    text_encoder_lora_alpha         = 4     
    train_text_encoder              = True  
    
    # Training params
    learning_rate                   = 1e-4
    text_encoder_lr                 = 5e-5  
    train_batch_size                = 1         # Aumente se tiver >12GB VRAM
    gradient_accumulation_steps     = 4         # Simula um batch maior (Batch efetivo = 1 * 4 = 4)
    num_train_epochs                = 6       
    resolution                      = 512
    
    # Otimização
    enable_xformers                 = True
    use_8bit_adam                   = True
    mixed_precision                 = "fp16"

    # Setup Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps = gradient_accumulation_steps,
        mixed_precision             = mixed_precision
    )
    device = accelerator.device

    # Carrega Modelos
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, 
        torch_dtype = torch.float16
    ).to(device)
    
    tokenizer, text_encoder, vae, unet = pipe.tokenizer, pipe.text_encoder, pipe.vae, pipe.unet

    # Otimizações de Memória
    unet.enable_gradient_checkpointing()
    if train_text_encoder:
        text_encoder.gradient_checkpointing_enable()
    
    if enable_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            print("✓ xFormers enabled")
        except:
            print("⚠ xFormers failed (install with: pip install xformers)")

    # Congela pesos base
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # VAE para CPU para economizar VRAM
    vae.to('cpu')
    torch.cuda.empty_cache()

    # Configura LoRA UNet
    unet_lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet.add_adapter(unet_lora_config)
    
    # Configura LoRA Text Encoder
    if train_text_encoder:
        text_encoder_lora_config = LoraConfig(
            r=text_encoder_lora_rank, lora_alpha=text_encoder_lora_alpha, init_lora_weights="gaussian",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"]
        )
        text_encoder.add_adapter(text_encoder_lora_config)
    
    # Garante que parâmetros treináveis estejam em float32 (necessário para estabilidade)
    for param in unet.parameters():
        if param.requires_grad: param.data = param.to(torch.float32)
    if train_text_encoder:
        for param in text_encoder.parameters():
            if param.requires_grad: param.data = param.to(torch.float32)

    # --- CARREGA DATASET ---
    print(f"Loading Dataset from {train_data_dir}...")
    train_dataset = CorelSDDataset(
        data_dir=train_data_dir,
        tokenizer=tokenizer,
        prompt_dict=CLASS_PROMPT_DICT,
        image_size=resolution
    )
    print(f"Total images: {len(train_dataset)}")

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
    if train_text_encoder:
        params_to_optimize += list(filter(lambda p: p.requires_grad, text_encoder.parameters()))

    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(params_to_optimize, lr=learning_rate, weight_decay=1e-2)
            print("✓ Using 8-bit AdamW")
        except ImportError:
            print("⚠ bitsandbytes missing. Using standard AdamW.")
            optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(params_to_optimize, lr=learning_rate, weight_decay=1e-2)

    lr_scheduler = get_scheduler("constant", optimizer=optimizer)

    # Prepare accelerator
    if train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # --- LOOP DE TREINO ---
    print("\nStarting training...")
    max_train_steps = num_train_epochs * len(train_dataloader)
    progress_bar = tqdm(range(max_train_steps), desc="Steps")

    for epoch in range(num_train_epochs):
        unet.train()
        if train_text_encoder: text_encoder.train()
        
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Encode Imagens (VAE)
                # Move VAE para GPU apenas agora
                vae.to(device)
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(device, dtype=torch.float16)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                vae.to('cpu') # Devolve para CPU

                # 2. Ruído e Timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 3. Encode Texto
                if train_text_encoder:
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                else:
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # 4. Predict e Loss
                target = noise # Para epsilon prediction (padrão SD 1.5)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # 5. Backprop
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                train_loss = loss.detach().item()
                progress_bar.set_postfix({"loss": f"{train_loss:.4f}"})
            
            # Limpeza de memória
            if step % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # --- SALVAR MODELO ---
    print("\nSaving LoRA weights...")
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(unet)))
        
        text_encoder_lora_layers = None
        if train_text_encoder:
            text_encoder = text_encoder.to(torch.float32)
            text_encoder_lora_layers = convert_state_dict_to_diffusers(get_peft_model_state_dict(accelerator.unwrap_model(text_encoder)))

        output_name = f"corel_lora_rank{lora_rank}_{formatted_date}.safetensors"
        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_layers,
            text_encoder_lora_layers=text_encoder_lora_layers,
            safe_serialization=True,
            weight_name=output_name
        )
        print(f"✓ Model saved to: {output_dir}/{output_name}")

if __name__ == "__main__":
    main()