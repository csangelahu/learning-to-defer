import torch 
import torch.nn.functional as F


# Eq 7 loss
# Note that g_y is not implemented here as a function of phi, though it could technically be done
class L2D_Equation7_Loss:
    def __init__(self, num_classes, num_experts):
        self.num_classes = num_classes
        self.num_experts = num_experts

    def forward(self, g_classes, g_perp, labels, expert_labels):
        """
        Args:
            g_classes: [N, K] classifier logits
            g_perp: [N, E] rejector logits (already conditioned on phi_e)
            labels: [N] ground truth labels
            expert_labels: [N, E] expert predictions for each sample
        """
        N, K = g_classes.shape
        E = g_perp.shape[1]
        device = g_classes.device
        labels = labels.long().to(device)

        # for numerical stability
        max_logit_class, _ = g_classes.max(dim=1, keepdim=True)  # [N, 1]
        max_logit_perp, _ = g_perp.max(dim=1, keepdim=True)      # [N, 1]
        max_all = torch.max(max_logit_class, max_logit_perp)     # [N, 1]
        g_classes_shifted = g_classes - max_all
        g_perp_shifted = g_perp - max_all

        # correct-class logit [N]
        g_y_shifted = g_classes_shifted[torch.arange(N, device=device), labels]
        
        # normalization constant Z
        exp_g_classes = torch.exp(g_classes_shifted)  # [N, K]
        sum_exp_classes = exp_g_classes.sum(dim=1)    # [N]
        exp_g_perp = torch.exp(g_perp_shifted)        # [N, E]
        Z = sum_exp_classes.unsqueeze(1) + exp_g_perp # [N, E]

        # Term 1
        term1 = -(g_y_shifted.unsqueeze(1) - torch.log(Z))  # [N, E]
        
        # Term 2
        indicator = (expert_labels.to(device) == labels.unsqueeze(1)).float()  # [N, E]
        term2 = -indicator * (g_perp_shifted - torch.log(Z))  # [N, E]
        
        # Sum over experts and average over batch
        loss = (term1 + term2).sum(dim=1).mean()
        return loss
    
# Eq 8 loss
class L2D_PopAvg_Loss:
    """
    Implements Eq. (8): population-averaged loss (shared deferral logit).
    Works with g_classes [N,K] and g_perp [N,E].
    """
    def __init__(self, num_classes, num_experts):
        self.num_classes = num_classes
        self.num_experts = num_experts

    def forward(self, g_classes, g_perp, labels, expert_labels):
        """
        Args:
            g_classes: [N, K] classifier logits
            g_perp: [N, E] rejector logits (one per expert)
            labels: [N]
            expert_labels: [N, E]
        """
        N, K = g_classes.shape
        E = g_perp.shape[1]
        device = g_classes.device
        labels = labels.to(device)

        # population-averaged deferral logit
        # g_perp = log( (1/E) * sum_e exp(g_perp_e) )
        g_perp = torch.logsumexp(g_perp, dim=1) - torch.log(torch.tensor(E, device=device, dtype=torch.float32))  # [N]
        exp_g_perp = torch.exp(g_perp)  # [N]

        exp_g_classes = torch.exp(g_classes)           # [N,K]
        Z = exp_g_classes.sum(dim=1) + exp_g_perp       # [N]

        # correct-class term
        g_y = g_classes[torch.arange(N), labels]       # [N]
        term1 = -torch.log(torch.exp(g_y) / Z)

        # deferral term
        frac_correct = (expert_labels == labels.unsqueeze(1)).float().mean(dim=1).to(device)  # [N]
        term2 = -frac_correct * torch.log(exp_g_perp / Z)

        loss = (term1 + term2).mean()
        return loss
