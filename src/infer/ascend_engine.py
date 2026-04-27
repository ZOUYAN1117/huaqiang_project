from infer.base_engine import BaseEngine


class AscendEngine(BaseEngine):
    def __init__(self, model_path: str):
        self.model_path = model_path
        # TODO: 后面接 ACL / ais_bench / Ascend 推理接口

    def infer(self, input_data):
        # TODO
        return None