import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import sys

# Importação nativa do Python 3.8+ para ler pacotes instalados
import importlib.metadata

from depth_anything_v2.dpt import DepthAnythingV2

def export_environment_logs():
    """Gera documentação automatizada das dependências e ativação para uso na Jetson."""
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Coleta os pacotes instalados no venv atual usando importlib.metadata
    try:
        installed_packages = sorted([
            f"{dist.metadata['Name']}=={dist.version}" 
            for dist in importlib.metadata.distributions()
        ])
    except Exception as e:
        installed_packages = [f"Erro ao ler pacotes: {str(e)}"]

    # Informações de Hardware e Ambiente
    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    python_version = sys.version.split()[0]
    
    # Comando de ativação utilizado pelo usuário
    activation_cmd = "source venv/da2/bin/activate"

    # --- SALVANDO ARQUIVO .TXT ---
    txt_path = os.path.join(log_dir, 'environment_log.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=====================================================\n")
        f.write("      DOCUMENTAÇÃO DE AMBIENTE - DEPTH ANYTHING V2   \n")
        f.write("=====================================================\n\n")
        f.write(f"Comando de Ativação Utilizado:\n> {activation_cmd}\n\n")
        f.write("Metadados do Sistema de Inferência:\n")
        f.write(f"- Python Versão: {python_version}\n")
        f.write(f"- Dispositivo de Hardware: {device_name}\n")
        f.write(f"- CUDA Runtime Versão: {cuda_version}\n\n")
        f.write("Bibliotecas Necessárias Instaladas (pip freeze local):\n")
        f.write("-----------------------------------------------------\n")
        for pkg in installed_packages:
            f.write(f"{pkg}\n")
            
    # --- SALVANDO ARQUIVO .YAML ---
    yaml_path = os.path.join(log_dir, 'environment_log.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write("environment_setup:\n")
        f.write(f"  activation_command: \"{activation_cmd}\"\n")
        f.write("  system_info:\n")
        f.write(f"    python_version: \"{python_version}\"\n")
        f.write(f"    hardware_device: \"{device_name}\"\n")
        f.write(f"    cuda_version: \"{cuda_version}\"\n")
        f.write("  required_libraries:\n")
        for pkg in installed_packages:
            f.write(f"    - \"{pkg}\"\n")

    print(f">>> Logs de ambiente salvos com sucesso na pasta '{log_dir}' (.txt e .yaml)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 - Dual Save')
    
    parser.add_argument('--img-path', type=str, required=True)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./outputs')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true')
    
    args = parser.parse_args()
    
    # Executa a geração de logs corrigida
    export_environment_logs()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    ckpt_path = f'checkpoints/depth_anything_v2_{args.encoder}.pth'
    depth_anything.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=True))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # --- BUSCA DE IMAGENS NA RAIZ DA PASTA ---
    if os.path.isfile(args.img_path):
        filenames = [args.img_path]
    else:
        extensions = ('*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG')
        filenames = []
        for ext in extensions:
            filenames.extend(glob.glob(os.path.join(args.img_path, ext)))
        filenames.sort()

    if not filenames:
        print(f"ERRO: Nenhuma imagem encontrada em: {os.path.abspath(args.img_path)}")
        exit()

    outdir_color = os.path.join(args.outdir, 'color')
    outdir_gray = os.path.join(args.outdir, 'grayscale')
    os.makedirs(outdir_color, exist_ok=True)
    os.makedirs(outdir_gray, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    print(f'>>> Processando {len(filenames)} imagens de {args.img_path}...')

    for k, filename in enumerate(filenames):
        raw_image = cv2.imread(filename)
        if raw_image is None: continue
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        depth_u8 = ((depth - depth.min()) / (depth.max() - depth.min()) * 255.0).astype(np.uint8)
        
        # Salvar Grayscale (Métricas)
        name = os.path.splitext(os.path.basename(filename))[0] + '.png'
        cv2.imwrite(os.path.join(outdir_gray, name), depth_u8)
        
        # Salvar Colorido (Plot)
        depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
        depth_color = (cmap(depth_norm)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
        if args.pred_only:
            cv2.imwrite(os.path.join(outdir_color, name), depth_color)
        else:
            split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
            cv2.imwrite(os.path.join(outdir_color, name), cv2.hconcat([raw_image, split_region, depth_color]))

print(f"\nFeito! Imagens em {args.outdir}")