from typing import Any


class Trajectory(object):
    def __init__(
        self,
        prompt: str,
        templated_prompt: str,
        padded_output_text: str,
        unpadded_output_text: str,
        score: float,
    ) -> None:
        self.prompt = prompt
        self.templated_prompt = templated_prompt
        self.padded_output_text = padded_output_text
        self.unpadded_output_text = unpadded_output_text
        self.score = score
        self.finished = self.unpadded_output_text != self.padded_output_text

    def get_json_representation(self, sparse: bool = True) -> dict[str, Any]:
        if sparse:
            return {
                "prompt": self.prompt,
                "output": self.unpadded_output_text,
                "score": self.score,
            }
        else:
            return {
                "prompt": self.prompt,
                "templated_prompt": self.templated_prompt,
                "padded_output_text": self.padded_output_text,
                "unpadded_output_text": self.unpadded_output_text,
                "score": self.score,
                "finished": self.finished,
            }

    def get_alpaca_representation(self, generator: str) -> dict[str, str]:
        return {
            "instruction": self.prompt,
            "output": self.unpadded_output_text,
            "generator": generator,
        }
