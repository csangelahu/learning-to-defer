import torch  # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore

class L2D_Equation7_Loss:
    def __init__(self, num_classes, num_experts):
        self.num_classes = num_classes
        self.num_experts = num_experts

    def forward(self, g_classes, g_perp, labels, expert_labels):
        N, K = g_classes.shape
        E = g_perp.shape[1]
        device = g_classes.device
        labels = labels.long().to(device)
        
        # Expand g_classes for each expert
        g_classes_exp = g_classes.unsqueeze(1).expand(N, E, K)  # [N, E, K]
        g_perp_exp = g_perp.unsqueeze(2)  # [N, E, 1]
        
        # Create (K+1)-dimensional logits for each expert
        logits = torch.cat([g_classes_exp, g_perp_exp], dim=2)  # [N, E, K+1]
        
        # Compute log softmax
        log_probs = F.log_softmax(logits, dim=2)  # [N, E, K+1]
        
        # Term 1: -log P(y|x) for each expert
        labels_exp = labels.unsqueeze(1).expand(N, E)  # [N, E]
        term1 = -log_probs.gather(2, labels_exp.unsqueeze(2)).squeeze(2)  # [N, E]
        
        # Term 2: -I[m_e = y] * log P(defer_e|x)
        expert_correct = (expert_labels == labels.unsqueeze(1)).float().to(device)  # [N, E]
        log_p_defer = log_probs[:, :, K]  # [N, E]
        term2 = -expert_correct * log_p_defer  # [N, E]
        
        # Total loss
        loss = (term1 + term2).mean()
        return loss
    
# Eq 8 loss
class L2D_PopAvg_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, g_classes, g_perp, labels, expert_labels):
        """
        g_classes: [N,K]
        g_perp: [N]
        labels: [N]
        expert_labels: [N,E]
        """
        N, K = g_classes.shape
        device = g_classes.device
        labels = labels.to(device)

        # Combine into [N, K+1] logits (K classes + 1 deferral)
        logits = torch.cat([g_classes, g_perp.unsqueeze(1)], dim=1)  # [N, K+1]
        log_probs = F.log_softmax(logits, dim=1)                     # [N, K+1]

        # Fraction of experts that predicted correctly
        frac_correct = (expert_labels == labels.unsqueeze(1)).float().mean(dim=1).to(device)  # [N]

        # -log P(y|x)
        term1 = -log_probs[torch.arange(N), labels]  # [N]

        # - frac_correct * log P(def|x)
        log_p_def = log_probs[:, K]  # last column = deferral class
        term2 = -frac_correct * log_p_def

        loss = (term1 + term2).mean()
        return loss