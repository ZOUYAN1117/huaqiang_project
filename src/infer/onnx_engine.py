from infer.base_engine import BaseEngine


class OnnxEngine(BaseEngine):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # TODO: 后面接 onnxruntime

    def infer(self, input_data):
        # TODO
        return None