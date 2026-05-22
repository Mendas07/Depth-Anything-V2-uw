import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np

# ==========================
# 1. CONFIGURAÇÃO DE CAMINHOS
# ==========================
# Ajustado para o caminho do seu Pop!_OS e DA2
BASE_PATH = Path("/home/pdi-b06/almacen/Depth-Anything-V2")
CSV_PATH = BASE_PATH / "outputs/metrics_results/no_ref_metrics_da2_final.csv"

OUT_DIR = BASE_PATH / "outputs/metrics_results/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLE_PATH = OUT_DIR / "benchmark_table_da2.png"

# ==========================
# 2. CARREGAR DADOS
# ==========================
if not CSV_PATH.exists():
    print(f"❌ Erro: Arquivo {CSV_PATH} não encontrado.")
    exit()

df = pd.read_csv(CSV_PATH)

# Identifica as colunas de métricas
metrics = ["Edge Align (↑)", "Smoothness (↓)", "EdgeAwareSmooth (↓)"]

# Função para truncar sem arredondar (mantém 6 casas fiéis)
def truncate_fixed(n, decimals=6):
    try:
        s = str(n)
        if '.' not in s:
            return s + '.' + '0' * decimals
        before, after = s.split('.')
        return before + '.' + after[:decimals].ljust(decimals, '0')
    except:
        return str(n)

# ==========================
# 3. GERAÇÃO DA TABELA (PNG)
# ==========================
df_table = df.copy()

# Aplica o truncamento rigoroso de 6 casas
for col in metrics:
    df_table[col] = df_table[col].apply(lambda x: truncate_fixed(x, 6))

# Ajuste da largura para acomodar os números
fig, ax = plt.subplots(figsize=(14, len(df_table) * 0.8 + 2))
ax.axis('off')

table = ax.table(
    cellText=df_table.values,
    colLabels=df_table.columns,
    loc='center',
    cellLoc='center'
)

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.8) # Escala vertical para dar respiro aos números

# Estilização Visual
for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#bdc3c7")
    if row == 0:
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor("#2c3e50") # Azul escuro 
    else:
        if row % 2 == 0:
            cell.set_facecolor("#f2f4f4")
        else:
            cell.set_facecolor("#ffffff")

# Linha de destaque por Dataset
for i in range(1, len(df_table)):
    if df_table.iloc[i]["Dataset"] != df_table.iloc[i-1]["Dataset"]:
        for col_idx in range(len(df_table.columns)):
            table[(i, col_idx)].set_linewidth(2.0)
            table[(i, col_idx)].set_edgecolor("black")

plt.title("Benchmarking: Depth Anything V2 (Underwater Datasets)", 
          fontsize=15, weight='bold', pad=20)

plt.savefig(TABLE_PATH, bbox_inches='tight', dpi=300)
plt.close()
print(f"✔️ Tabela salva: {TABLE_PATH}")

# ==========================
# 4. GRÁFICOS COMPARATIVOS
# ==========================
for metric in metrics:
    plt.figure(figsize=(10, 6))
    
    # Ordem lógica dos modelos para o gráfico
    model_order = {"Small": 0, "Base": 1, "Large": 2}
    
    for dataset in df["Dataset"].unique():
        df_ds = df[df["Dataset"] == dataset].copy()
        df_ds["Order"] = df_ds["Model"].map(model_order)
        df_ds = df_ds.sort_values("Order")

        plt.plot(
            df_ds["Model"], 
            df_ds[metric], 
            marker='s', # Quadrado para diferenciar do DA3
            linewidth=2,
            markersize=8,
            label=dataset
        )

    plt.title(f"DA2 Comparison - {metric}", fontsize=14, weight='bold')
    plt.xlabel("Model Scale", fontsize=12)
    plt.ylabel("Value (6-dec precision)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Datasets", frameon=True)

    # Nome do arquivo limpo
    clean_name = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('↑', 'up').replace('↓', 'down')
    plt.savefig(OUT_DIR / f"da2_plot_{clean_name}.png", bbox_inches="tight", dpi=300)
    plt.close()

print(f"✅ Todos os plots foram gerados em: {OUT_DIR}")