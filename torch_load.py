import torch
from depth_anything_v2.dpt import DepthAnythingV2

# ============================================================
# CONFIGURAÇÃO DO MODELO
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

encoder = "vitb"  # vits | vitb | vitl | vitg

model_configs = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384]
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768]
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024]
    },
    "vitg": {
        "encoder": "vitg",
        "features": 384,
        "out_channels": [1536, 1536, 1536, 1536]
    }
}

# ============================================================
# CRIA MODELO
# ============================================================

model = DepthAnythingV2(**model_configs[encoder])

# ============================================================
# LOAD DOS PESOS
# ============================================================

checkpoint_path = "checkpoints/depth_anything_v2_vitb.pth"

state_dict = torch.load(
    checkpoint_path,
    map_location=DEVICE
)

model.load_state_dict(state_dict)

model = model.to(DEVICE)

# ============================================================
# MODO EVAL
# ============================================================

model.eval()

# ============================================================
# INPUT DUMMY
# ============================================================

dummy_input = torch.randn(
    1,      # batch
    3,      # RGB
    518,    # altura
    518     # largura
).to(DEVICE)

# ============================================================
# EXPORTAÇÃO ONNX
# ============================================================

torch.onnx.export(
    model,
    dummy_input,
    "depth_anything_v2.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {
            0: "batch_size",
            2: "height",
            3: "width"
        },
        "output": {
            0: "batch_size",
            2: "height",
            3: "width"
        }
    }
)

print("Modelo exportado para ONNX com sucesso!")