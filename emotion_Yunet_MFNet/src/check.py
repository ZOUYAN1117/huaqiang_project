import onnx

for path in [
    "face_detection_yunet_2023mar.onnx",
    "facial_expression_recognition_mobilefacenet_2022july.onnx"
]:
    print("\n===", path, "===")
    model = onnx.load(path)

    for inp in model.graph.input:
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else d.dim_param)
        print("input:", inp.name, dims)

    for out in model.graph.output:
        dims = []
        for d in out.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.dim_value > 0 else d.dim_param)
        print("output:", out.name, dims)