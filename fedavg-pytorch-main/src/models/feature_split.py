# FedFed-style Feature Split Module: separate h into z_s (sensitive, shared) and z_r (robust, local).
# Lightweight "information bottleneck": z_s low-dim for sharing; z_r residual for local use.
# Plug-in only; does not depend on specific model architecture.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureSplitModule(nn.Module):
    """
    Splits intermediate feature h into:
      - z_s: performance-sensitive feature (low-dim, for sharing across clients)
      - z_r: performance-robust feature (residual, local only)

    Uses gate + low-dim projection for information bottleneck (FedFed idea).
    """

    def __init__(self, feature_dim, sensitive_dim):
        super(FeatureSplitModule, self).__init__()
        self.feature_dim = feature_dim
        self.sensitive_dim = sensitive_dim

        # Low-dim projection: h -> z_s (shared)
        self.proj_s = nn.Linear(feature_dim, sensitive_dim)

        # Gate: sigmoid(MLP(h)) to control how much of h goes to z_s
        self.gate_mlp = nn.Sequential(
            nn.Linear(feature_dim, max(feature_dim // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(feature_dim // 4, 1), 1),
        )

        # Reverse projection: z_s (low-dim) -> same space as h for residual
        self.proj_rev = nn.Linear(sensitive_dim, feature_dim)

    def forward(self, h):
        """
        Args:
            h: (B, feature_dim) intermediate representation
        Returns:
            z_s: (B, sensitive_dim) performance-sensitive feature (for sharing)
            z_r: (B, feature_dim) residual / robust feature (local only)
        """
        # z_s low-dim
        z_s_lowdim = self.proj_s(h)  # (B, sensitive_dim)
        gate = torch.sigmoid(self.gate_mlp(h))  # (B, 1)
        z_s_in_h_space = gate * self.proj_rev(z_s_lowdim)  # (B, feature_dim)
        z_r = h - z_s_in_h_space  # residual

        return z_s_lowdim, z_r
