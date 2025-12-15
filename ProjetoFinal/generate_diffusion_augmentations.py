#%%
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os
import glob
from datetime import datetime

# --- CONFIGURAÇÕES ---
OUTPUT_DIR = "./diffusion_augmentations"
LORA_DIR_OUTPUT = "./corel_lora_output"
WIDTH = 400
HEIGHT = 400
BASE_SEED = 42
GUIDANCE_SCALE = 7.5
NUM_IMAGES_PER_CLASS = 20

# Atualizei a lista para incluir explicitamente o ID da classe
# para garantir que o nome do arquivo comece com "1_", "2_", etc.
PROMPTS_TO_GENERATE = [
    {"id": "0001", "prompt": "a photorealistic photo of the british royal guard"},
    {"id": "0002", "prompt": "a photorealistic photo of a steam train"},
    {"id": "0003", "prompt": "a photorealistic photo of a pie"},
    {"id": "0004", "prompt": "a photorealistic photo of different vegetables"},
    {"id": "0005", "prompt": "a photorealistic photo of a snowy landscape"},
    {"id": "0006", "prompt": "a photorealistic photo of a orange sunset"},
]

NEGATIVE_PROMPT = "low quality, blur, watermark, text, bad anatomy, deformed, ugly, pixelated, extra limbs"

# --- 1. PREPARAÇÃO ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Encontrar LoRA
print("=" * 50)
print(f"Buscando LoRA em {LORA_DIR_OUTPUT}...")
safetensors_files = glob.glob(os.path.join(LORA_DIR_OUTPUT, "*.safetensors"))

if not safetensors_files:
    safetensors_files = glob.glob(os.path.join(LORA_DIR_OUTPUT, "*.bin"))
    if not safetensors_files:
        if os.path.isdir(LORA_DIR_OUTPUT):
             lora_path = LORA_DIR_OUTPUT
             lora_name = "folder_format"
        else:
             raise FileNotFoundError("Nenhum modelo LoRA encontrado.")
    else:
        lora_path = max(safetensors_files, key=os.path.getmtime)
        lora_name = os.path.basename(lora_path)
else:
    lora_path = max(safetensors_files, key=os.path.getmtime)
    lora_name = os.path.basename(lora_path)

print(f"✓ Modelo encontrado: {lora_name}")

# --- 2. CARREGAR PIPELINE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Carregando Stable Diffusion em {device}...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None 
).to(device)

pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

# Otimizações
pipe.enable_attention_slicing(slice_size="auto")
pipe.enable_vae_tiling()
pipe.enable_model_cpu_offload()

print("Carregando pesos do LoRA...")
try:
    if os.path.isfile(lora_path):
        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_name)
    else:
        pipe.load_lora_weights(lora_path)
    print("✓ LoRA carregado.")
except Exception as e:
    print(f"Erro ao carregar LoRA: {e}")

# --- 3. LOOP DE GERAÇÃO EM MASSA ---
print("\n" + "=" * 50)
print(f"GERANDO {NUM_IMAGES_PER_CLASS} IMAGENS POR CLASSE")
print(f"Formato de salvamento: {{ID}}_aug_{{indice}}.png")
print("=" * 50)

total_generated = 0

for idx, item in enumerate(PROMPTS_TO_GENERATE):
    prompt = item["prompt"]
    class_id = item["id"] # Ex: "1", "2"...
    
    print(f"\n[{idx+1}/{len(PROMPTS_TO_GENERATE)}] Gerando Classe ID: {class_id}")
    
    for i in range(NUM_IMAGES_PER_CLASS):
        # Semente dinâmica para garantir variedade
        current_seed = BASE_SEED + i + (idx * 100)
        
        image = pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            guidance_scale=GUIDANCE_SCALE,
            width=WIDTH,
            height=HEIGHT,
            generator=torch.Generator(device).manual_seed(current_seed)
        ).images[0]
        
        # --- LÓGICA DE NOMECLATURA ---
        # Salva como: 1_aug_000.png, 1_aug_001.png, etc.
        # Isso satisfaz filename.startswith(f"{class_id}_")
        filename = f"{class_id}_aug_{i:03d}.png"
        
        save_path = os.path.join(OUTPUT_DIR, filename)
        image.save(save_path)
        
        print(f"    -> Salvo: {filename} (Seed: {current_seed})")
        total_generated += 1

print("\n" + "=" * 50)
print(f"CONCLUÍDO! Total de {total_generated} imagens salvas em '{OUTPUT_DIR}'")