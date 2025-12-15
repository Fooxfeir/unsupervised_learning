#%%
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import textwrap

# --- CONFIGURAÇÕES ---
OUTPUT_DIR = "./generated_corel_images"
LORA_DIR_OUTPUT = "./corel_lora_output"
WIDTH = 512
HEIGHT = 512
SEED = 42
GUIDANCE_SCALE = 7.5

# Prompts (agora o código adicionará as aspas automaticamente na visualização)
PROMPTS_TO_GENERATE = [
    {"filename": "class_1_guard", "prompt": "a photorealistic photo of the british royal guard"},
    {"filename": "class_2_train", "prompt": "a photorealistic photo of a steam train"},
    {"filename": "class_3_pie",   "prompt": "a photorealistic photo of a pie"},
    {"filename": "class_4_veg",   "prompt": "a photorealistic photo of vegetables"},
    {"filename": "class_5_snow",  "prompt": "a photorealistic photo of a snowy landscape"},
    {"filename": "class_6_sunset","prompt": "a photorealistic photo of a orange sunset"},
]

NEGATIVE_PROMPT = "low quality, blur, watermark, text, bad anatomy, deformed, ugly, pixelated"

# Cria diretório
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 1. ENCONTRAR O MODELO LORA ---
print("=" * 50)
print(f"Searching for LoRA in {LORA_DIR_OUTPUT}...")
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

print(f"✓ Found latest model: {lora_name}")

# --- 2. CARREGAR PIPELINE ---
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

print(f"Loading LoRA weights...")
try:
    if os.path.isfile(lora_path):
        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_name)
    else:
        pipe.load_lora_weights(lora_path)
    print("✓ LoRA loaded successfully")
except Exception as e:
    print(f"Erro ao carregar LoRA: {e}")

# --- 3. GERAÇÃO ---
generated_images = []
generated_prompts = []

print("\n" + "=" * 50)
print("STARTING GENERATION")
print("=" * 50)

for idx, item in enumerate(PROMPTS_TO_GENERATE):
    prompt = item["prompt"]
    fname = item["filename"]
    
    print(f"\n[{idx+1}/{len(PROMPTS_TO_GENERATE)}] Generating: {fname}")
    
    image = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=30,
        guidance_scale=GUIDANCE_SCALE,
        width=WIDTH,
        height=HEIGHT,
        generator=torch.Generator(device).manual_seed(SEED)
    ).images[0]
    
    generated_images.append(image)
    generated_prompts.append(prompt)
    
    save_path = os.path.join(OUTPUT_DIR, f"{fname}_{timestamp}.png")
    image.save(save_path)
    
    torch.cuda.empty_cache()

print("\n" + "=" * 50)
print("Creating annotated grid...")

# Ajustei figsize para (16, 10). Isso é grande o suficiente para qualidade, 
# mas menor que o anterior, gerando menos pixels totais.
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()

# Adiciona título principal
fig.suptitle(f"Geração Stable Diffusion + LoRA (Corel Dataset) - Seed {SEED}", fontsize=16, fontweight='bold', y=0.98)

for i, ax in enumerate(axes):
    if i < len(generated_images):
        ax.imshow(generated_images[i])
        
        # Aspas no prompt
        full_prompt = f'"{generated_prompts[i]}"'
        
        # Quebra de linha ajustada para o novo tamanho (width=40)
        wrapped_prompt = "\n".join(textwrap.wrap(full_prompt, width=40))
        
        # Fonte levemente menor para caber melhor
        ax.set_title(wrapped_prompt, fontsize=10, style='italic', pad=10)
        
        ax.axis('off')
    else:
        ax.axis('off')

plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.02, wspace=0.1, hspace=0.25)

# --- CORREÇÃO DO ERRO ---
# Removemos 'optimize=True' que estava causando o crash.
# Mantemos 'bbox_inches' e 'dpi' que são nativos do Matplotlib.
# 'quality' geralmente é aceito, mas se der erro novamente, remova a linha 'quality=85'.

grid_filename = f"grid_with_prompts_{timestamp}.jpg"
grid_path = os.path.join(OUTPUT_DIR, grid_filename)

try:
    plt.savefig(
        grid_path, 
        dpi=150,              # 150 é excelente para Overleaf
        format='jpg',         # Formato comprimido
        bbox_inches='tight',  # Corta bordas brancas
        quality=85            # Qualidade do JPG (se der erro aqui também, apague essa linha)
    )
except TypeError:
    # Fallback se a versão do seu Matplotlib for muito antiga e não aceitar 'quality'
    print("Versão antiga do Matplotlib detectada. Salvando com parâmetros padrão...")
    plt.savefig(
        grid_path, 
        dpi=150, 
        format='jpg', 
        bbox_inches='tight'
    )

print(f"✓ Grid OTIMIZADO salvo em: {grid_path}")

# Mostra o gráfico
plt.show()

print("=" * 50)
print("Done!")