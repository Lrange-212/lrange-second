import onnx
import onnx_graphsurgeon as gs
import numpy as np

# 加载 ONNX 模型
model_path = 'seal.onnx'
onnx_model = onnx.load(model_path)

# 使用 Polygraphy 进行常量折叠
folded_model = fold_constants(onnx_model)

# 使用 onnx_graphsurgeon 将 TopK 的 k 转换为常量
graph = gs.import_onnx(onnx_model)
#graph = gs.import_onnx(folded_model)
for node in graph.nodes:
    if node.op == 'TopK' :
        print(node)
        k_input = node.inputs[1]
        if k_input.inputs and isinstance(k_input.inputs[0], gs.ir.node.Node):
            identity_node = k_input.inputs[0]
            node.inputs[1] = identity_node.inputs[0]

# 导出修改后的模型
modified_model_path = 'seal1.onnx'
onnx.save(gs.export_onnx(graph), modified_model_path)
#onnx.save(folded_model, modified_model_path)

# 检查模型
onnx.checker.check_model(modified_model_path)
print("Model checked successfully!")
