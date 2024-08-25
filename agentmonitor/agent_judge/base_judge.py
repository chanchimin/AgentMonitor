from typing import Dict

class BaseJudge:
    def __init__(self, config: Dict):
        self.config = config

    def judging(self, input_path, output_path, use_name):
        raise NotImplementedError