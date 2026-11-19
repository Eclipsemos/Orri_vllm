from dataclasses import dataclass

@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_token: int = 64
    ignore_eos: bool = False

    def __post_int__(self):
        assert self.temperature > 1e-10, "greedy sampling is forbidden"