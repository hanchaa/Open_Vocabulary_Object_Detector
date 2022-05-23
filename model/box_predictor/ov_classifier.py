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
                 temperature=0.07
                 ):
        super().__init__()
        input_dim = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        self.projection = nn.Linear(input_dim, prompt_dim)

        prompt_embedding = torch.tensor(np.load(prompt_path), dtype=torch.float, device="cuda")
        self.prompt_embedding = normalize(prompt_embedding, dim=0)
        assert self.prompt_embedding.shape[0] == prompt_dim
        assert self.prompt_embedding.shape[1] == num_classes

        self.background_embedding = torch.randn((prompt_dim, 1), requires_grad=True, device="cuda")

        self.temperature = temperature

    def forward(self, x):
        x = self.projection(x)
        classifier = torch.cat([self.prompt_embedding, normalize(self.background_embedding, dim=0)], dim=1)

        return (x @ classifier) / self.temperature
