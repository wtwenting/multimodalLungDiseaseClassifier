from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from monai.networks.nets import DenseNet121


DEFAULT_CT_CHECKPOINT = Path(
    "/Users/wenting/Projects/PsuprProj/CT/model/"
    "ct_3d_covid_normal_densenet121.pt"
)


class FrozenCTEncoder(nn.Module):
    def __init__(self, checkpoint=DEFAULT_CT_CHECKPOINT):
        super().__init__()

        model = DenseNet121(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
        )

        state = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        model.load_state_dict(state, strict=True)

        self.features = model.features

        for parameter in self.features.parameters():
            parameter.requires_grad = False

        self.features.eval()

    def train(self, mode=True):
        super().train(mode)
        self.features.eval()
        return self

    def forward(self, ct):
        with torch.no_grad():
            features = self.features(ct)
            features = F.relu(features)
            features = F.adaptive_avg_pool3d(features, 1)
            return torch.flatten(features, 1)