import torch 
import torch.nn.functional as F
import time 

class L2D_Loss:

    def __init__(self, num_classes):
        """
        Initialize the L2D_Loss class.

        Args:
            num_classes (int): Number of classes (excluding the deferral class).
        """
        self.expected_dim = num_classes + 1

    def check_dimensions(self, outputs):

        """
        Check if the tensor has the expected number of classes.
        Raises ValueError if not.

        Args:
            outputs (Tensor): Model output tensor of shape (batch_size, self.expected_dim).
        """

        if  outputs.size(1) != self.expected_dim:
            raise ValueError(f"Output dimension mismatch. Expected {self.expected_dim}, got {outputs.size(1)}.")
        
    def loss_fn(self, logits, true_labels, expert_labels, deferral_class_index, param_type, alpha=0.5):

        """
        Returns specified loss.
        Raises ValueError if type of parametrization is unknown.

        Args:
            logits (Tensor): output logits from the model.
            true_labels (Tensor): Ground-truth labels.
            expert_labels (Tensor): Labels provided by the expert.
            deferral_class_index (int): Index of the deferral class in logits.
            param_type (str): Type of loss to use: ('softmax', 'asymmetric_sm', 'one_vs_all', 'realizable_sm').
            alpha (float, optional): Weighting factor for realizable loss. Default is 0.5.

        Returns: loss value.
        """

        self.check_dimensions(logits)

        if param_type == "softmax":
            return self.loss_softmax(logits, true_labels, expert_labels, deferral_class_index)
        elif param_type == "asymmetric_sm":
            return self.loss_asym_sm(logits, true_labels, expert_labels, deferral_class_index)
        elif param_type == "one_vs_all":
            return self.loss_one_v_all(logits, true_labels, expert_labels, deferral_class_index)
        elif param_type == "realizable_sm":
            return self.loss_realizable_sm(logits, true_labels, expert_labels, deferral_class_index, alpha)
        else:
            raise ValueError(f"Unknown param_type: {param_type}")
        
    def loss_softmax(self, logits, true_labels, expert_labels, deferral_class_index):

        """
        Standard softmax loss with deferral.

        Args:
            logits (Tensor): Raw model outputs.
            true_labels (Tensor): Ground-truth labels.
            expert_labels (Tensor): Expert-provided labels.
            deferral_class_index (int): Index of the deferral class.

        Returns: loss value.
        """

        logits_stable = logits - torch.max(logits, dim=1, keepdim=True).values # numerical stability trick 
        
        exp_logits = torch.exp(logits_stable)
        sum_exp_logits = torch.sum(exp_logits, dim=1, keepdim=True)
        softmax_probs = exp_logits / sum_exp_logits

        true_labels = true_labels.long()
        gathered_probs = softmax_probs[torch.arange(softmax_probs.size(0)), true_labels]
        loss = -(torch.log(gathered_probs))

        indicator = (true_labels == expert_labels).float()
        expert_probs = softmax_probs[torch.arange(softmax_probs.size(0)), deferral_class_index]
        loss -= (indicator * torch.log(expert_probs))

        return torch.mean(loss)
        
    @staticmethod
    def phi(x):
        """
        Returns (log(1 + exp(-x)))

        Args:
            x (Tensor): Input tensor.
        """

        return F.softplus(-x)  # log(1 + exp(-x))

    def loss_one_v_all(self, logits, true_labels, expert_labels, deferral_class_index):
        """
        One-vs-all loss with deferral.

        Args:
            logits (Tensor): Model output logits.
            true_labels (Tensor): Ground-truth class labels.
            expert_labels (Tensor): Expert-provided labels.
            deferral_class_index (int): Index of the deferral class.

        Returns: loss value.
        """

        batch_size, num_classes = logits.shape

        g_y = logits[torch.arange(batch_size), true_labels]
        loss = L2D_Loss.phi(g_y)

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[torch.arange(batch_size), true_labels] = False  # exclude true class
        mask[torch.arange(batch_size), deferral_class_index] = False  # exclude deferral class
        loss += L2D_Loss.phi(-logits[mask]).view(batch_size, -1).sum(dim=1)

        g_deferral = logits[:, deferral_class_index]
        loss += L2D_Loss.phi(-g_deferral)

        indicator = (expert_labels == true_labels).float()
        loss += indicator * (L2D_Loss.phi(g_deferral) - L2D_Loss.phi(-g_deferral))

        return loss.mean()

    def loss_asym_sm(self, logits, true_labels, expert_labels, deferral_class_index):

        """
        Asymmetric softmax loss with deferral.

        Args:
            logits (Tensor): Model output logits.
            true_labels (Tensor): Ground-truth class labels.
            expert_labels (Tensor): Expert-provided labels.
            deferral_class_index (int): Index of the deferral class.

        Returns: loss value.
        """
        
        batch_size, num_classes = logits.shape

        logits_stable = logits - torch.max(logits, dim=1, keepdim=True).values # numerical stability trick
        exp_logits = torch.exp(logits_stable)

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, deferral_class_index] = False  # Exclude deferral class

        sum_exp_logits_K = torch.sum(exp_logits * mask, dim=1, keepdim=True)  # sum over K classes
        sum_exp_logits_K1 = torch.sum(exp_logits, dim=1, keepdim=True)  # sum over all classes

        phi_A_SM = torch.zeros_like(logits)

        phi_A_SM[:, :deferral_class_index] = exp_logits[:, :deferral_class_index] / sum_exp_logits_K  # before deferral
        phi_A_SM[:, deferral_class_index + 1:] = exp_logits[:, deferral_class_index + 1:] / sum_exp_logits_K  # after deferral

        masked_logits = exp_logits.masked_fill(~mask, float('-inf'))
        max_masked_logits = torch.max(masked_logits, dim=1, keepdim=True).values
        phi_A_SM[:, deferral_class_index] = exp_logits[:, deferral_class_index] / (
            sum_exp_logits_K1.squeeze(-1) - max_masked_logits.squeeze(-1)
        )

        true_labels = true_labels.long()
        gathered_probs = phi_A_SM[torch.arange(batch_size), true_labels]
        loss = -torch.log(gathered_probs)

        indicator_m_not_y = (expert_labels != true_labels).float()
        indicator_m_eq_y = (expert_labels == true_labels).float()

        loss -= indicator_m_not_y * torch.log(1 - phi_A_SM[:, deferral_class_index])
        loss -= indicator_m_eq_y * torch.log(phi_A_SM[:, deferral_class_index])

        return loss.mean()
    
    def loss_realizable_sm(self, logits, true_labels, expert_labels, deferral_class_index, alpha):
        """
        Realizable softmax loss with deferral.

        Args:
            logits (Tensor): Model output logits.
            true_labels (Tensor): Ground-truth class labels.
            expert_labels (Tensor): Expert-provided labels.
            deferral_class_index (int): Index of the deferral class.
            alpha (float): Weight for expert-influenced term.

        Returns: loss value.
        """

        batch_size, num_classes = logits.shape

        logits_stable = logits - torch.max(logits, dim=1, keepdim=True).values # numerical stability trick

        exp_logits = torch.exp(logits_stable)
        sum_exp_logits_all = torch.sum(exp_logits, dim=1, keepdim=True)

        true_labels_y = true_labels.long()
        expert_labels_h = expert_labels.long()

        # Term 1
        indicator_h_eq_y = (expert_labels_h == true_labels_y).float()
        term1_numerator = exp_logits[torch.arange(batch_size), true_labels_y] + indicator_h_eq_y * exp_logits[torch.arange(batch_size), deferral_class_index]
        term1_denominator = sum_exp_logits_all.squeeze(1)
        loss1 = -alpha * torch.log(term1_numerator / term1_denominator)

        # Term 2

        mask = torch.ones_like(logits, dtype=torch.bool)
        mask[:, deferral_class_index] = False

        sum_exp_logits_no_deferral = torch.sum(exp_logits * mask, dim=1, keepdim=True).squeeze(1)  # sum over K classes

        term2_numerator = exp_logits[torch.arange(batch_size), true_labels_y]
        term2_denominator = sum_exp_logits_no_deferral
        loss2 = -(1 - alpha) * torch.log(term2_numerator / term2_denominator)

        loss = loss1 + loss2

        return loss.mean()