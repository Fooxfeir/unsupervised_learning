#%%
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
from diffusers.utils import make_image_grid
from datetime import datetime
import os
import glob

# --- CONFIGURAÇÕES ---
# 1. ESCOLHA A CLASSE QUE VOCÊ TREINOU (Deve ser a mesma do treino)
TARGET_CLASS_ID = "0001" 

# Configurações de Geração
NUM_IMAGES = 4        # Quantas imagens gerar
WIDTH = 512
HEIGHT = 512
SEED = 42             # Mude para variar os resultados
GUIDANCE_SCALE = 7.5

# Caminho dinâmico (pega a pasta criada pelo treino daquela classe)
LORA_DIR_OUTPUT = f"./corel_class_{TARGET_CLASS_ID}_output"
OUTPUT_DIR = f"./generated_class_{TARGET_CLASS_ID}"

# Dicionário de Prompts (O mesmo do treino)
CLASS_PROMPT_DICT = {
    "0001": "a photorealistic photo of the british royal guard",
    "0002": "a photorealistic photo of a steam train",
    "0003": "a photorealistic photo of a pie",
    "0004": "a photorealistic photo of different vegetables",
    "0005": "a photorealistic photo of a snowy landscape",
    "0006": "a photorealistic photo of a orange sunset",
}

NEGATIVE_PROMPT = "low quality, blur, watermark, text, bad anatomy, deformed, ugly, pixelated"

# --- 1. PREPARAÇÃO ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Pega o prompt correto
if TARGET_CLASS_ID in CLASS_PROMPT_DICT:
    PROMPT = CLASS_PROMPT_DICT[TARGET_CLASS_ID]
else:
    raise ValueError(f"Classe '{TARGET_CLASS_ID}' não encontrada no dicionário!")

print("=" * 50)
print(f"GENERATING FOR CLASS: {TARGET_CLASS_ID}")
print(f"Prompt: {PROMPT}")
print(f"Model Folder: {LORA_DIR_OUTPUT}")
print("=" * 50)

# --- 2. ENCONTRAR O MODELO ---
safetensors_files = glob.glob(os.path.join(LORA_DIR_OUTPUT, "*.safetensors"))

if not safetensors_files:
    raise FileNotFoundError(f"Nenhum modelo encontrado em {LORA_DIR_OUTPUT}. Você treinou a classe {TARGET_CLASS_ID}?")

lora_path = max(safetensors_files, key=os.path.getmtime)
lora_name = os.path.basename(lora_path)
print(f"✓ Found model: {lora_name}")

# --- 3. CARREGAR PIPELINE ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Stable Diffusion on {device}...")

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

# Carregar LoRA
print(f"Loading LoRA weights...")
pipe.load_lora_weights(LORA_DIR_OUTPUT, weight_name=lora_name)
print("✓ LoRA loaded successfully")

# --- 4. GERAÇÃO EM LOOP ---
generated_images = []

print(f"\nGerando {NUM_IMAGES} imagens...")

for i in range(NUM_IMAGES):
    # Variar a seed para cada imagem
    current_seed = SEED + i
    
    print(f"  [{i+1}/{NUM_IMAGES}] Seed: {current_seed}")
    
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        guidance_scale=GUIDANCE_SCALE,
        width=WIDTH,
        height=HEIGHT,
        generator=torch.Generator(device).manual_seed(current_seed)
    ).images[0]
    
    generated_images.append(image)
    
    # Salvar
    save_name = f"class_{TARGET_CLASS_ID}_{timestamp}_{i+1}.png"
    save_path = os.path.join(OUTPUT_DIR, save_name)
    image.save(save_path)
    
    torch.cuda.empty_cache()

# --- 5. GRID FINAL ---
print("\nCriando grid de resumo...")
if len(generated_images) > 0:
    rows = (len(generated_images) + 1) // 2
    grid = make_image_grid(generated_images, cols=2, rows=rows)
    grid_path = os.path.join(OUTPUT_DIR, f"grid_class_{TARGET_CLASS_ID}_{timestamp}.png")
    grid.save(grid_path)
    print(f"✓ Grid salvo em: {grid_path}")

print("=" * 50)
print("Geração Concluída!")