import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ContextGatingModule(nn.Module):
    """
    Cross-modal gating block that learns channel-wise and spatial masks
    for each modality conditioned on the concatenated PET/CT features.
    """

    def __init__(
        self,
        dim: int,
        reduction: int = 4,
        min_channel: int = 32,
        residual: bool = True,
    ) -> None:
        super().__init__()
        hidden = max(dim // max(reduction, 1), min_channel)
        self.residual = residual

        # channel-wise gates (operate on pooled context)
        self.channel_gate_rgb = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
        )
        self.channel_gate_pet = nn.Sequential(
            nn.Conv2d(dim * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, dim, kernel_size=1, bias=True),
        )

        # spatial gates (captures local context differences)
        self.spatial_gate_rgb = nn.Sequential(
            nn.Conv2d(dim * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.spatial_gate_pet = nn.Sequential(
            nn.Conv2d(dim * 2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid(),
        )
        self.act = nn.Sigmoid()

    def _compute_gate(self, x_primary: torch.Tensor, x_secondary: torch.Tensor, channel_gate, spatial_gate) -> torch.Tensor:
        """
        Produce a modulation mask for x_primary conditioned on both inputs.
        """
        fused = torch.cat([x_secondary, x_primary], dim=1)
        channel_mask = self.act(channel_gate(
            F.adaptive_avg_pool2d(fused, output_size=1)))
        spatial_mask = spatial_gate(fused)
        # broadcast channel mask to spatial resolution and blend with spatial mask
        gate = channel_mask * spatial_mask
        return gate

    def forward(self, x_rgb: torch.Tensor, x_pet: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gate_rgb = self._compute_gate(
            x_rgb, x_pet, self.channel_gate_rgb, self.spatial_gate_rgb)
        gate_pet = self._compute_gate(
            x_pet, x_rgb, self.channel_gate_pet, self.spatial_gate_pet)

        if self.residual:
            x_rgb = x_rgb * (1.0 + gate_rgb)
            x_pet = x_pet * (1.0 + gate_pet)
        else:
            x_rgb = x_rgb * gate_rgb
            x_pet = x_pet * gate_pet
        return x_rgb, x_pet