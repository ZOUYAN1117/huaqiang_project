import onnx

src = "facial_expression_recognition_mobilefacenet_2022july.onnx"
dst = "facial_expression_recognition_mobilefacenet_2022july_fixed.onnx"

model = onnx.load(src)

# 找到所有 initializer（权重）
initializer_names = {init.name for init in model.graph.initializer}

# 只保留真正的数据输入（去掉权重）
new_inputs = []
for inp in model.graph.input:
    if inp.name not in initializer_names:
        new_inputs.append(inp)

# 替换 input
del model.graph.input[:]
model.graph.input.extend(new_inputs)

# 校验并保存
onnx.checker.check_model(model)
onnx.save(model, dst)

print("saved:", dst)
print("real inputs:")
for inp in model.graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else d.dim_param)
    print(inp.name, dims)