import numpy as np
import torch
from detectron2.layers import ShapeSpec
from torch import nn
from torch.nn.functional import normalize


class OpenVocabularyClassifier(nn.Module):
    def __init__(self,
                 input_shape: ShapeSpec,
                 num_classes: int,
                 prompt_path: str,
                 prompt_dim: int,
                 temperature=0.07,
                 use_binary_ce=False
                 ):
        super().__init__()
        input_dim = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.projection = nn.Linear(input_dim, prompt_dim)

        prompt_embedding = torch.tensor(np.load(prompt_path), dtype=torch.float, device="cuda")
        self.prompt_embedding = normalize(prompt_embedding, dim=-1)
        assert self.prompt_embedding.shape[0] == num_classes
        assert self.prompt_embedding.shape[1] == prompt_dim

        self.background_embedding = nn.Parameter(torch.randn((1, prompt_dim))) if not use_binary_ce else None

        self.temperature = temperature

    def forward(self, x):
        x = self.projection(x)
        x = normalize(x, dim=-1)

        if self.background_embedding is not None:
            classifier = torch.cat([self.prompt_embedding, normalize(self.background_embedding, dim=-1)], dim=0)

        else:
            classifier = self.prompt_embedding

        return (x @ classifier.T) / self.temperature
