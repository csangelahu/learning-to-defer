import torch
import torch.nn.functional as F

class L2D_MultiExpertLoss:
    def __init__(self, num_classes, num_experts):
        """
        Initialize the L2D_MultiExpertLoss class.

        Args:
            num_classes (int): Number of standard classes.
            num_experts (int): Number of experts.
        """
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.expected_dim = num_classes + num_experts

    def check_dimensions(self, logits, expert_labels):
        """
        Check that the logits and expert_labels have the expected shape.
        """
        if logits.size(1) != self.expected_dim:
            raise ValueError(
                f"Output dimension mismatch. Expected {self.expected_dim}, got {logits.size(1)}."
            )
        if expert_labels.size(1) != self.num_experts:
            raise ValueError(
                f"Expert label mismatch. Expected {self.num_experts} experts, got {expert_labels.size(1)}."
            )

    @staticmethod
    def phi(x):
        """Return φ(x) = log(1 + exp(-x))."""
        return F.softplus(-x)

    def loss_fn(self, logits, labels, expert_labels, param_type="softmax"):
        """
        Dispatch to the correct loss function.

        Args:
            logits (Tensor): Model output logits of shape (batch_size, K + J).
            labels (Tensor): Ground-truth labels of shape (batch_size,).
            expert_labels (Tensor): Predictions from experts of shape (batch_size, J).
            param_type (str): Which loss to compute ("softmax" or "one_vs_all").
        """
        self.check_dimensions(logits, expert_labels)

        if param_type == "softmax":
            return self.loss_softmax_multi_expert(logits, labels, expert_labels)
        elif param_type == "one_vs_all":
            return self.loss_ova_multi_expert(logits, labels, expert_labels)
        else:
            raise ValueError(f"Unknown param_type: {param_type}")

    def loss_softmax_multi_expert(self, logits, labels, expert_labels):
        """
        Multi-expert softmax loss.
        """

        labels = labels.long()
        log_probs = F.log_softmax(logits, dim=1)

        # Term 1
        log_prob_y = log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        loss_term1 = -log_prob_y

        # Term 2
        indicator_mask = (expert_labels == labels.unsqueeze(1)).float()
        log_probs_experts = log_probs[:, self.num_classes:]  # last J columns
        expert_term = (indicator_mask * log_probs_experts).sum(dim=1)
        loss_term2 = -expert_term

        total_loss = loss_term1 + loss_term2
        return total_loss.mean()


    def loss_ova_multi_expert(self, logits, labels, expert_labels):
        """
        One-vs-All (OvA) multi-expert loss.
        """
        batch_size = logits.size(0)

        labels = labels.long()
        class_logits = logits[:, :self.num_classes]
        expert_logits = logits[:, self.num_classes:]

        # Term 1
        g_y = class_logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        term1 = self.phi(g_y)

        # Term 2
        y_one_hot = F.one_hot(labels, num_classes=self.num_classes)
        incorrect_class_mask = (y_one_hot == 0)
        term2 = self.phi(-class_logits[incorrect_class_mask]).view(batch_size, -1).sum(dim=1)

        # Term 3
        term3 = self.phi(-expert_logits).sum(dim=1)

        # Term 4
        indicator_mask = (expert_labels == labels.unsqueeze(1)).float()
        phi_g_expert = self.phi(expert_logits)
        phi_neg_g_expert = self.phi(-expert_logits)
        diff_term = phi_g_expert - phi_neg_g_expert
        term4 = (indicator_mask * diff_term).sum(dim=1)

        total_loss = term1 + term2 + term3 + term4
        return total_loss.mean()