"""
Inspect ONNX model input/output tensor names and shapes.
Usage: uv run inspect_shapes.py
"""

import os
import sys

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

MODELS = [
    "duration_predictor.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx",
]


def inspect_onnx(model_name):
    path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(path):
        print(f"  Not found: {path} — run download.py first")
        return

    try:
        import onnx
        from onnx import shape_inference, TensorProto
    except ImportError:
        print("Install onnx: uv pip install onnx")
        sys.exit(1)

    model = onnx.load(path)
    graph = model.graph

    def dtype_name(t):
        return TensorProto.DataType.Name(t)

    def fmt_shape(shape):
        if shape is None:
            return "?"
        dims = []
        for d in shape.dim:
            dims.append(str(d.dim_value) if d.dim_value else (d.dim_param or "?"))
        return "[" + ", ".join(dims) + "]"

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"  IR version: {model.ir_version}")
    print(f"  Opset: {model.opset_import[0].version}")
    print(f"  Nodes: {len(graph.node)}")

    print(f"\n  Inputs:")
    for inp in graph.input:
        shape = inp.type.tensor_type.shape
        dtype = inp.type.tensor_type.elem_type
        print(f"    {inp.name:40s} {dtype_name(dtype):8s} {fmt_shape(shape)}")

    print(f"\n  Outputs:")
    for out in graph.output:
        shape = out.type.tensor_type.shape
        dtype = out.type.tensor_type.elem_type
        print(f"    {out.name:40s} {dtype_name(dtype):8s} {fmt_shape(shape)}")


def main():
    for model in MODELS:
        inspect_onnx(model)
    print()


if __name__ == "__main__":
    main()
