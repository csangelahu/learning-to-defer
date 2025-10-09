import torch
import numpy as np

class PopulationSimulator:
    def __init__(self, num_experts, num_classes, overlap_prob, oracle_classes_per_expert, seed=None):
        self.num_experts = num_experts
        self.num_classes = num_classes
        self.overlap_prob = overlap_prob
        self.oracle_classes_per_expert = oracle_classes_per_expert
        self.rng = np.random.RandomState(seed)
        self.latest_psi = None

    def sample_expert_probabilities(self):
        """returns [num_experts, num_classes] torch tensor"""
        psi = np.zeros((self.num_experts, self.num_classes))
        
        for e in range(self.num_experts):
            oracle_classes = self.rng.choice(self.num_classes, 
                                        self.oracle_classes_per_expert, 
                                        replace=False)
            
            for k in range(self.num_classes):
                if k in oracle_classes:
                    psi[e, k] = 1.0
                else:
                    concentration = 1000
                    alpha = self.overlap_prob * concentration
                    beta = (1 - self.overlap_prob) * concentration
                    psi[e, k] = self.rng.beta(alpha, beta)
    
        self.latest_psi = torch.tensor(psi, dtype=torch.float32)   
        return self.latest_psi

    def sample_predictions_given_phi(self, labels, psi=None):
        """
        Simulate expert predictions given psi.
        labels: tensor or array of shape [N]
        returns expert_preds tensor [N, E] of ints in 0..K-1
        """
        if psi is None:
            assert self.latest_psi is not None, "No psi sampled yet"
            psi = self.latest_psi

        labels_np = labels.cpu().numpy()
        N = len(labels_np)
        E, K = psi.shape
        preds = np.zeros((N, E), dtype=np.int64)

        for e in range(E):
            psi_e = psi[e].cpu().numpy()  # [K]
            probs = psi_e[labels_np]      # [N]
            rand = self.rng.rand(N)
            correct_mask = rand < probs
            preds[:, e][correct_mask] = labels_np[correct_mask]
            incorrect_idx = np.nonzero(~correct_mask)[0]
            for idx in incorrect_idx:
                true = labels_np[idx]
                choices = list(range(K))
                choices.remove(true)
                preds[idx, e] = self.rng.choice(choices)
        return torch.tensor(preds, dtype=torch.long)

