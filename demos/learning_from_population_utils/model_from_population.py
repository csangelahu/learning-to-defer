import torch
import torch.nn as nn

"""
This file is intended for use in the provided demos and is not part of the core library.
"""

# Transforms images into vectors 
class SimpleCNNFeatureExtractor(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        flat = 64 * 7 * 7
        self.fc = nn.Sequential(nn.Linear(flat, 512), nn.ReLU(), nn.Linear(512, out_dim))

    def forward(self, x):
        if x.ndim == 2:
            x = x.view(-1, 1, 28, 28)
        return self.fc(self.conv(x))


# Expert encoder : transform into representation of expert, psi 
class ExpertEmbeddingEncoder(nn.Module):
    def __init__(self, num_classes, phi_dim=8, hidden=[64, 32]):
        super().__init__()
        layers = []
        in_dim = num_classes
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, phi_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, psi_all):
        # psi_all: [E, K] -> phi_all: [E, phi_dim]
        return self.mlp(psi_all)

# Combines classifier with rejector
class ExpertDecisionModule(nn.Module):
    def __init__(self, feature_dim, num_classes, phi_dim=8, num_experts=3,
                 classifier_hidden=[256], rejector_hidden=[128]):
        super().__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts

        # classifier head
        cls_layers = []
        in_dim = feature_dim
        for h in classifier_hidden:
            cls_layers += [nn.Linear(in_dim, h), nn.ReLU()]; in_dim = h
        cls_layers.append(nn.Linear(in_dim, num_classes))
        self.classifier_head = nn.Sequential(*cls_layers)

        # rejector head
        rej_layers = []
        in_dim = feature_dim + phi_dim
        for h in rejector_hidden:
            rej_layers += [nn.Linear(in_dim, h), nn.ReLU()]; in_dim = h
        rej_layers.append(nn.Linear(in_dim, 1))
        self.rejector_head = nn.Sequential(*rej_layers)

    def forward(self, f_x, phi_all):
        """f_x: [N, feature_dim], phi_all: [E, phi_dim]"""
        g_classes = self.classifier_head(f_x)  # [N, K]
        N = f_x.size(0); E = phi_all.size(0)
        phi = phi_all.unsqueeze(0).expand(N, E, -1)    # [N,E,phi_dim]
        f_expand = f_x.unsqueeze(1).expand(-1, E, -1)  # [N,E,feature_dim]
        fe = torch.cat([f_expand, phi], dim=2)         # [N,E,feature+phi]
        fe_flat = fe.view(N * E, -1)
        g_perp_flat = self.rejector_head(fe_flat)      # [N*E, 1]
        g_perp = g_perp_flat.view(N, E)                # [N, E]
        return g_classes, g_perp                        # [N, K], [N, E]
