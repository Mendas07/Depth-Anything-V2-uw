import numpy as np
import pandas as pd
import cv2
import os
from pathlib import Path

# ==============================================================================
# CONFIG - Caminhos Atualizados (Pop!_OS / RTX 4070 Ti)
# ==============================================================================
BASE_PATH = Path("/home/pdi-b06/almacen/Depth-Anything-V2")
DATASETS_ROOT = BASE_PATH / "datasets"
OUTPUTS_ROOT = BASE_PATH / "outputs"

# Mapeamento completo baseado no seu 'ls' de outputs
datasets_config = {
    "SUIM": {
        "rgb_ref": DATASETS_ROOT / "suim",
        "models": {
            "Small": OUTPUTS_ROOT / "da2-suim-small/grayscale",
            "Base":  OUTPUTS_ROOT / "da2-suim-base/grayscale",
            "Large": OUTPUTS_ROOT / "da2-suim-large/grayscale",
        }
    },
    "UIEB": {
        "rgb_ref": DATASETS_ROOT / "uieb",
        "models": {
            "Small": OUTPUTS_ROOT / "da2-uieb-small/grayscale",
            "Base":  OUTPUTS_ROOT / "da2-uieb-base/grayscale",
            "Large": OUTPUTS_ROOT / "da2-uieb-large/grayscale",
        }
    },
    "USIS10K": {
        "rgb_ref": DATASETS_ROOT / "usis10k",
        "models": {
            "Small": OUTPUTS_ROOT / "da2-usis10k-small/grayscale",
            "Base":  OUTPUTS_ROOT / "da2-usis10k-base/grayscale",
            "Large": OUTPUTS_ROOT / "da2-usis10k-large/grayscale",
        }
    }
}

# Configuração de Saída
OUT_DIR = OUTPUTS_ROOT / "metrics_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "no_ref_metrics_da2_final.csv"

# ==============================================================================
# FUNÇÕES DE MÉTRICAS
# ==============================================================================

def edge_alignment_score(rgb, depth):
    """Mede o alinhamento das bordas da profundidade com a imagem RGB."""
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges_rgb = cv2.Canny(rgb_gray, 50, 150)
    
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    if grad_mag.max() < 1e-6: return 0.0
    grad_mag = grad_mag / grad_mag.max()
    edges_depth = (grad_mag > 0.1)

    intersection = np.logical_and(edges_rgb > 0, edges_depth).sum()
    union = np.logical_or(edges_rgb > 0, edges_depth).sum()
    return float(intersection / (union + 1e-8))

def depth_smoothness(depth):
    """Mede a variação local (suavidade) do mapa de profundidade."""
    dx = np.abs(np.diff(depth, axis=1))
    dy = np.abs(np.diff(depth, axis=0))
    return float((dx.mean() + dy.mean()) / 2.0)

def edge_aware_smoothness(rgb, depth):
    """Mede a suavidade ponderada pelas bordas da imagem RGB."""
    rgb_gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    dx_depth = np.abs(np.diff(depth, axis=1))
    dy_depth = np.abs(np.diff(depth, axis=0))
    dx_img = np.abs(np.diff(rgb_gray, axis=1))
    dy_img = np.abs(np.diff(rgb_gray, axis=0))
    
    weight_x = np.exp(-dx_img)
    weight_y = np.exp(-dy_img)
    return float((dx_depth * weight_x).mean() + (dy_depth * weight_y).mean()) / 2.0

# ==============================================================================
# LOOP DE PROCESSAMENTO
# ==============================================================================

results = []

for ds_name, ds in datasets_config.items():
    # Coleta arquivos originais (JPG ou PNG)
    img_files = sorted(list(ds["rgb_ref"].glob("*.png")) + list(ds["rgb_ref"].glob("*.jpg")) + list(ds["rgb_ref"].glob("*.jpeg")))
    
    if not img_files:
        print(f"⚠️ Atenção: Nenhuma imagem encontrada em {ds['rgb_ref']}")
        continue

    print(f"\n📂 Analisando {ds_name} ({len(img_files)} imagens)")

    for model_name, depth_dir in ds["models"].items():
        if not depth_dir.exists():
            print(f"   ❌ Pasta não encontrada: {depth_dir}")
            continue

        metrics_stack = []

        for img_path in img_files:
            # Busca a predição (mesmo nome, mas forçando .png que o opencv salva)
            depth_file = depth_dir / (img_path.stem + ".png")

            if not depth_file.exists():
                continue

            # Carregamento e Conversão
            rgb = cv2.imread(str(img_path))
            if rgb is None: continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_GRAYSCALE)
            if depth_img is None: continue
            depth = depth_img.astype(np.float32) / 255.0

            # Redimensiona profundidade para bater com RGB original se necessário
            if rgb.shape[:2] != depth.shape[:2]:
                depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Cálculo das métricas
            e = edge_alignment_score(rgb, depth)
            s = depth_smoothness(depth)
            eas = edge_aware_smoothness(rgb, depth)

            metrics_stack.append([e, s, eas])

        if metrics_stack:
            avg = np.mean(metrics_stack, axis=0)
            results.append([ds_name, model_name, avg[0], avg[1], avg[2]])
            print(f"   ✅ {model_name}: Edge Align: {avg[0]:.4f} | Smooth: {avg[1]:.4f}")

# ==============================================================================
# RESULTADOS FINAIS
# ==============================================================================

df = pd.DataFrame(results, columns=[
    "Dataset", "Model", "Edge Align (↑)", "Smoothness (↓)", "EdgeAwareSmooth (↓)"
])

print("\n" + "="*90)
print("📊 RESULTADOS DESEMPENHO - DEPTH ANYTHING V2 (NO-REFERENCE METRICS)")
print("="*90)
print(df.sort_values(by=["Dataset", "Model"]).to_string(index=False))
print("="*90)

df.to_csv(OUT_PATH, index=False)
print(f"\n💾 Relatório salvo com sucesso em: {OUT_PATH}")