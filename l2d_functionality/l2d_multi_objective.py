import torch
import torch.nn.functional as F
import numpy as np

class MultipleConstraintDGNP:
    """
    Implements single-constraint d-GNP for Expert Intervention Budget
    Usage:
      - Call fit_on_validation to estimate k_hat and p_hat.
      - Call apply_to_loader(test_loader) to get post-processed (final_preds, raw_class_preds, defer_idx)
    
    Notes:
      - psi1(x) = [0,...,0,1]
    """

    def __init__(self, model: torch.nn.Module, num_classes: int, device = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        self.d = num_classes + 1
        self.k_hat = None
        self.p_hat = None
  
        self._psi0_val = None  # (n_val, d)
        self._psi1_val = None  # (n_val, d)

    @staticmethod
    def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
        # logits: (B, num_classes)
        return F.softmax(logits, dim=1)

    def gather_validation_psis(self, val_loader, deferral_class_index: int):
        """
        Compute psi0_hat and psi1_hat on validation set.
        Returns (psi0_hat, psi1_hat).
        psi0_hat: (N, d) where last column is expert_correct_indicator (0/1)
        psi1_hat: (N, d) with zeros except last column = 1
        """
        self.model.eval()
        psi0_list = []
        expert_correct_list = []
        with torch.no_grad():
            for images, labels, expert_labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                expert_labels = expert_labels.to(self.device)

                outputs = self.model(images) 
                class_logits = outputs[:, :self.num_classes]  # (B, L)
                class_probs = self.softmax_probs(class_logits).cpu().numpy()  # (B, L)

                if expert_labels.dim() > 1 and expert_labels.size(1) > 1:
                    # if expert_labels holds multiple experts per sample, choose first column (for now)
                    expert_lab = expert_labels[:, 0].squeeze()
                else:
                    expert_lab = expert_labels.squeeze()

                expert_correct = (expert_lab == labels).long().cpu().numpy()  # (B,)

                # psi0: [class_probs..., expert_correct]
                B = class_probs.shape[0]
                psi0 = np.zeros((B, self.d), dtype=float)
                psi0[:, :self.num_classes] = class_probs
                psi0[:, -1] = expert_correct.astype(float)
                psi0_list.append(psi0)

                expert_correct_list.append(expert_correct)

        psi0_hat = np.vstack(psi0_list)  # (N, d)

        psi1_hat = np.zeros_like(psi0_hat)
        psi1_hat[:, -1] = 1.0
        return psi0_hat, psi1_hat

    @staticmethod
    def tau_single(k: float, psi0: np.ndarray, psi1: np.ndarray) -> np.ndarray:
        """
        Returns one-hot argmax of scores = psi0 - k*psi1 with tie policy.
        tie policy (Theorem 4.2): if multiple maximizers I exist -> choose i that minimizes psi1 and j that maximizes psi0
        """
        scores = psi0 - k * psi1  # (n, d)
        n, d = scores.shape
        f = np.zeros_like(scores)
        maxvals = scores.max(axis=1, keepdims=True)
        is_max = np.isclose(scores, maxvals, atol=1e-12)
        for idx in range(n):
            I = np.nonzero(is_max[idx])[0]
            if I.size == 1:
                choice = I[0]
            else:
                # choose index in I with minimal psi1
                psi1_vals = psi1[idx, I]
                min_idx = np.argmin(psi1_vals)
                candidates = I[psi1_vals == psi1_vals[min_idx]]
                if candidates.size == 1:
                    choice = candidates[0]
                else:
                    # break tie by maximizing psi0 among candidates
                    psi0_vals = psi0[idx, candidates]
                    choice = candidates[np.argmax(psi0_vals)]
            f[idx, choice] = 1.0
        return f

    def C_hat(self, k: float, psi0: np.ndarray, psi1: np.ndarray) -> float:
        """
        C(k) = mean_i < f_{k,0}(x_i), psi1(x_i) >
        """
        f = self.tau_single(k, psi0, psi1)
        val = (f * psi1).sum(axis=1).mean()
        return float(val)

    def fit_on_validation(self, val_loader, deferral_class_index: int, delta: float,
                          grid_min: float = -10.0, grid_max: float = 10.0, grid_size: int = 2000):
        """
        Fit k and p on validation set to satisfy deferral budget delta.
        Returns (k_hat, p_hat).
        """
        psi0_hat, psi1_hat = self.gather_validation_psis(val_loader, deferral_class_index)
        self._psi0_val = psi0_hat
        self._psi1_val = psi1_hat

        ks = np.linspace(grid_min, grid_max, grid_size)
        C_vals = np.array([self.C_hat(k, psi0_hat, psi1_hat) for k in ks])

        feasible_indices = np.where(C_vals <= delta + 1e-12)[0]
        if feasible_indices.size == 0:
            raise ValueError("No feasible k found on grid. Expand grid range.")
        idx = feasible_indices[0]
        k_star = float(ks[idx])
        C_k = float(C_vals[idx])

        if idx > 0:
            C_left = float(C_vals[idx - 1])
        else:
            C_left = float(self.C_hat(k_star - (ks[1] - ks[0]) * 0.5, psi0_hat, psi1_hat))

        if abs(C_left - C_k) <= 1e-3:
            p = 0.0
        else:
            numerator = C_k - delta
            denom = C_k - C_left
            if abs(denom) < 1e-12:
                p = 0.0
            else:
                p = float(np.clip(numerator / denom, 0.0, 1.0))

        self.k_hat = k_star
        self.p_hat = p
        
        # print(f"[d-GNP budget] k_hat={self.k_hat:.6g}, p_hat={self.p_hat:.6g}, C(k)={C_k:.6g}, C_left={C_left:.6g}")
        
        return self.k_hat, self.p_hat

    def compute_f_k_p_for_psi(self, psi0: np.ndarray) -> np.ndarray:
        """
        Given psi0 rows for a batch, compute s(x) = f_{k_hat,p_hat}(x) probabilities.
        """

        assert (self.k_hat is not None) and (self.p_hat is not None), "Must call fit_on_validation first"
        n = psi0.shape[0]
        psi1 = np.zeros_like(psi0); psi1[:, -1] = 1.0

        f_k0 = self.tau_single(self.k_hat, psi0, psi1)

        if self.p_hat == 0.0:
            return f_k0 # no randomization 

        # locate rows with multiple maxima and randomize as in theorem 4.2
        scores = psi0 - self.k_hat * psi1
        maxvals = scores.max(axis=1, keepdims=True)
        is_max = np.isclose(scores, maxvals, atol=1e-12)
        f = f_k0.copy()
        for idx in range(n):
            I = np.nonzero(is_max[idx])[0]
            if I.size <= 1:
                continue
            # i = index that minimizes psi, j = index that maximizes psi0
            psi1_vals = psi1[idx, I] 
            i_idx = I[np.argmin(psi1_vals)]
            psi0_vals = psi0[idx, I]
            j_idx = I[np.argmax(psi0_vals)]
            p = self.p_hat
            f[idx, :] = 0.0
            f[idx, i_idx] = p
            f[idx, j_idx] = 1.0 - p
        return f

    def apply_to_loader(self, loader, deferral_class_index: int, use_expert_at_test: bool = True):
        """
        Apply post-processed rule on a loader.
        Returns (final_preds, raw_class_preds, defer_idx, true_labels, expert_labels_if_available)
        """

        assert (self.k_hat is not None) and (self.p_hat is not None), "Must call fit_on_validation first"
        self.model.eval()
        all_final_preds = []
        all_raw_preds = []
        all_defer_idx = []
        all_labels = []
        all_expert_labels = []

        with torch.no_grad():
            for images, labels, expert_labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                expert_labels = expert_labels.to(self.device)

                outputs = self.model(images)
                class_logits = outputs[:, :self.num_classes]
                class_probs = F.softmax(class_logits, dim=1).cpu().numpy()     # (B, L)
                B = class_probs.shape[0]

                # Build psi0
                # if use_expert_at_test=True, use the provided expert_labels to compute correctness
                if use_expert_at_test:
                    if expert_labels.dim() > 1 and expert_labels.size(1) > 1:
                        expert_lab = expert_labels[:, 0].squeeze()
                    else:
                        expert_lab = expert_labels.squeeze()
                    expert_correct = (expert_lab == labels).long().cpu().numpy()
                else:
                    # assume expert is correct
                    expert_correct = np.ones(B, dtype=float)

                psi0 = np.zeros((B, self.d), dtype=float)
                psi0[:, :self.num_classes] = class_probs
                psi0[:, -1] = expert_correct.astype(float)

                s_probs = self.compute_f_k_p_for_psi(psi0)  # (B, d)
                classifier_scores = s_probs[:, :self.num_classes]
                defer_scores = s_probs[:, -1]
                h_star = classifier_scores.argmax(axis=1)  # predicted class by classifier if not deferred
                classifier_max = classifier_scores[np.arange(B), h_star]
                r_star = (defer_scores > classifier_max).astype(int)  # 1 if defer

                final_pred = h_star.copy()
                if use_expert_at_test:
                    final_pred[r_star == 1] = expert_lab.cpu().numpy()[r_star == 1]
                else:
                    final_pred[r_star == 1] = h_star[r_star == 1]

                all_final_preds.append(torch.from_numpy(final_pred))
                all_raw_preds.append(torch.from_numpy(h_star))
                all_defer_idx.append(torch.from_numpy(r_star))
                all_labels.append(labels.cpu())
                all_expert_labels.append(expert_labels.cpu())

        all_final_preds = torch.cat(all_final_preds)
        all_raw_preds = torch.cat(all_raw_preds)
        all_defer_idx = torch.cat(all_defer_idx)
        all_labels = torch.cat(all_labels)
        all_expert_labels = torch.cat(all_expert_labels)
        return all_final_preds, all_raw_preds, all_defer_idx, all_labels, all_expert_labels
