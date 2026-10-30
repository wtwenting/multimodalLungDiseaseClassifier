import torch

from pretrained_encoders import FrozenCTEncoder


encoder = FrozenCTEncoder()

encoder.eval()

print("Encoder loaded successfully")

print(
    "Trainable parameters:",
    sum(
        p.numel()
        for p in encoder.parameters()
        if p.requires_grad
    )
)

with torch.no_grad():
    dummy_ct = torch.randn(1, 1, 64, 128, 128)

    features = encoder(dummy_ct)

print("Feature shape:", features.shape)