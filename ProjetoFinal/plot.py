import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. ÁREA DE CONFIGURAÇÃO MANUAL (INSIRA SEUS VALORES AQUI)
# ==============================================================================
# Substitua os valores abaixo pelos resultados obtidos nos seus testes.
# Formato: [ARI, NMI, Silhouette]

VALORES = {
    "Original (Spatial)": {
        "Base":      [0.609, 0.706, 0.450],  # [ARI, NMI, Sil]
        "Augmented": [0.603, 0.625, 0.487]   # [ARI, NMI, Sil]
    },
    "Modificada (Semantic)": {
        "Base":      [0.789, 0.817, 0.579],  # [ARI, NMI, Sil]
        "Augmented": [0.542, 0.621, 0.451]   # [ARI, NMI, Sil]
    }
}

METRICAS = ["ARI", "NMI", "Silhouette"]

# Configurações Visuais
COR_ORIGINAL = "#1f77b4"   # Azul
COR_MODIFICADA = "#ff7f0e" # Laranja
TEXTURA_AUG = "///"        # Padrão de hachura para Augmented
TEXTURA_BASE = ""          # Sem textura para Base

# ==============================================================================
# 2. GERAÇÃO DO GRÁFICO
# ==============================================================================
def plotar_graficos():
    # Configuração do Grid
    x = np.arange(len(METRICAS))  # Posições no eixo X (0, 1, 2)
    width = 0.20                  # Largura de cada barra
    
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Extração de Dados ---
    # Original
    orig_base = VALORES["Original (Spatial)"]["Base"]
    orig_aug  = VALORES["Original (Spatial)"]["Augmented"]
    # Modificada
    mod_base  = VALORES["Modificada (Semantic)"]["Base"]
    mod_aug   = VALORES["Modificada (Semantic)"]["Augmented"]

    # --- Plotagem das Barras ---
    
    # Grupo 1: Original (Esquerda do centro)
    rects1 = ax.bar(x - 1.5*width, orig_base, width, label='Original - Base', 
                    color=COR_ORIGINAL, hatch=TEXTURA_BASE, edgecolor='white')
    
    rects2 = ax.bar(x - 0.5*width, orig_aug, width, label='Original - Aug', 
                    color=COR_ORIGINAL, hatch=TEXTURA_AUG, edgecolor='white', alpha=0.9)

    # Grupo 2: Modificada (Direita do centro)
    rects3 = ax.bar(x + 0.5*width, mod_base, width, label='Modificada - Base', 
                    color=COR_MODIFICADA, hatch=TEXTURA_BASE, edgecolor='white')
    
    rects4 = ax.bar(x + 1.5*width, mod_aug, width, label='Modificada - Aug', 
                    color=COR_MODIFICADA, hatch=TEXTURA_AUG, edgecolor='white', alpha=0.9)

    # --- Estilização ---
    ax.set_ylabel('Pontuação (Score)')
    ax.set_title('Comparação de Métricas de Clusterização: DGAE Original vs Modificado')
    ax.set_xticks(x)
    ax.set_xticklabels(METRICAS, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.05) # Assume que métricas vão de 0 a 1
    
    # Grid de fundo para facilitar leitura
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    # --- Adicionar Rótulos de Valor em cima das barras ---
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, rotation=0)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    autolabel(rects4)

    # --- Legenda Customizada ---
    # Criamos uma legenda que explica as Cores e outra as Texturas para ficar limpo
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor=COR_ORIGINAL, label='Original'),
        Patch(facecolor=COR_MODIFICADA, label='Modificada'),
        Patch(facecolor='gray', hatch=TEXTURA_BASE, label='Dataset Base', alpha=0.5),
        Patch(facecolor='gray', hatch=TEXTURA_AUG, label='Dataset Aumentado', alpha=0.5)
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', ncol=2, fontsize=10)

    plt.tight_layout()
    
    # Salvar e Mostrar
    filename = "comparacao_metricas_dgae.png"
    plt.savefig(filename, dpi=300)
    print(f"Gráfico salvo como: {filename}")
    plt.show()

if __name__ == "__main__":
    plotar_graficos()