import onnx
import onnxruntime as ort
import numpy as np

# ============================================================
# VERIFICA ONNX
# ============================================================

model_path = "depth_anything_v2.onnx"

onnx_model = onnx.load(model_path)

onnx.checker.check_model(onnx_model)

print("ONNX VALIDADO!")

# ============================================================
# CRIA SESSÃO
# ============================================================

ort_session = ort.InferenceSession(
    model_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

print("ONNX Runtime carregado!")

# ============================================================
# INPUT
# ============================================================

dummy_input = np.random.randn(
    1,
    3,
    518,
    518
).astype(np.float32)

# ============================================================
# INFERÊNCIA
# ============================================================

outputs = ort_session.run(
    None,
    {"input": dummy_input}
)

print("Inferência executada!")

print("Shape saída:")
print(outputs[0].shape)