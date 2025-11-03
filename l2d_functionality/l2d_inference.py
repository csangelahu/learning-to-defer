import torch
from tqdm import tqdm


class L2DInference:
    """Unified inference class for standard and population-based learning-to-defer."""

    def __init__(self, num_classes, model=None, cnn=None, l2d=None,
                 expert_pop=None, device='cpu', expert_aware=None, tau=None, realizable_sm=False):
        """
        Args:
            num_classes (int): Number of classes.
            model (nn.Module): Used for standard L2D models (non-pop).
            cnn (nn.Module): Feature extractor for learning from pop models.
            l2d (nn.Module): Expert decision module for learning from pop.
            expert_pop: PopulationSimulator instance (for population-based L2D).
            device: Torch device.
            expert_aware (bool): True for Eq.7 (expert-aware), False for Eq.8 (pop-avg).
            tau (float, optional): Threshold for the deferral decision (realizable softmax).
            realizable_sm (bool, optional): Whether tau-based deferral rule is applied.
        """
        self.model = model
        self.cnn = cnn
        self.l2d = l2d
        self.num_classes = num_classes
        self.expert_pop = expert_pop
        self.device = device
        self.expert_aware = expert_aware
        self.tau = tau
        self.realizable_sm = realizable_sm

    def run(self, test_loader, psi_test=None, expert_idx=None):
        if self.model is not None:
            return self._run_standard(test_loader)
        else:
            return self._run_population(test_loader, psi_test, expert_idx)

    def _run_standard(self, test_loader):
        """Handles standard L2D inference (non-population)."""
        self.model.eval()
        all_final_preds, all_raw_preds, all_defer_idx, all_labels, all_expert_labels = [], [], [], [], []

        with torch.no_grad():
            for images, labels, expert_labels in test_loader:
                if expert_labels.dim() == 1:
                    expert_labels = expert_labels.unsqueeze(1)

                outputs = self.model(images)
                logits_classes = outputs[:, :self.num_classes]
                deferral_logits = outputs[:, self.num_classes:]

                _, raw_pred = torch.max(logits_classes, dim=1)

                if self.realizable_sm and self.tau is not None:
                    max_logits, _ = torch.max(logits_classes, dim=1)
                    defer_mask = (deferral_logits.squeeze() - max_logits) >= self.tau
                    full_pred = raw_pred.clone()
                    full_pred[defer_mask] = self.num_classes
                else:
                    _, full_pred = torch.max(outputs, dim=1)
                    defer_mask = full_pred >= self.num_classes

                defer_indices = torch.full_like(raw_pred, 0)

                if defer_mask.any():
                    defer_expert_idx = full_pred[defer_mask] - self.num_classes + 1
                    defer_indices[defer_mask] = defer_expert_idx
                    deferred_samples = torch.nonzero(defer_mask).squeeze(1)
                    chosen_expert_labels = expert_labels[
                        deferred_samples, full_pred[defer_mask] - self.num_classes
                    ]
                    corrected_pred = full_pred.clone()
                    corrected_pred[deferred_samples] = chosen_expert_labels
                else:
                    corrected_pred = raw_pred.clone()

                all_final_preds.append(corrected_pred)
                all_raw_preds.append(raw_pred)
                all_defer_idx.append(defer_indices)
                all_labels.append(labels)
                all_expert_labels.append(expert_labels)

        all_final_preds = torch.cat(all_final_preds)
        all_raw_preds = torch.cat(all_raw_preds)
        all_defer_idx = torch.cat(all_defer_idx)
        all_labels = torch.cat(all_labels)
        all_expert_labels = torch.cat(all_expert_labels)

        return all_final_preds, all_raw_preds, all_defer_idx, all_labels, all_expert_labels
        
    def _run_population(self, test_loader, psi_test=None, expert_idx=0):
        """Handles population-based inference (Eq.7 and Eq.8 variants)."""
        if self.cnn is not None:
            self.cnn.eval()
        self.l2d.eval()

        if psi_test is None:
            psi_test = self.expert_pop.sample_expert_probabilities()

        all_final_preds, all_labels, all_raw_preds, all_expert_labels, all_defer_idx = [], [], [], [], []

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Eval (expert {expert_idx})"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                # Sample expert predictions
                expert_preds = self.expert_pop.sample_predictions_given_phi(
                    labels.cpu(), psi_test[expert_idx:expert_idx+1].cpu()
                )[:, 0].to(self.device)

                if self.expert_aware:
                    f_x = self.cnn(imgs)
                    # Obtain the class logits and expert logits
                    g_classes, g_perp_all = self.l2d(f_x, psi_test.to(self.device))
                    g_perp = g_perp_all[:, expert_idx]
                else:
                    g_classes, g_perp = self.l2d(imgs)

                logits = torch.cat([g_classes, g_perp.unsqueeze(1)], dim=1)
                

                max_class_logits, class_preds = torch.max(g_classes, dim=1)
                defer_mask = g_perp > max_class_logits
                defer_indices = torch.zeros_like(class_preds)
                defer_indices[defer_mask] = 1

                final_preds = class_preds.clone()
                final_preds[defer_mask] = expert_preds[defer_mask]

                all_final_preds.append(final_preds.cpu())
                all_raw_preds.append(class_preds.cpu())
                all_defer_idx.append(defer_indices.cpu())
                all_labels.append(labels.cpu())
                all_expert_labels.append(expert_preds.cpu())

        all_final_preds = torch.cat(all_final_preds)
        all_labels = torch.cat(all_labels)
        all_raw_preds = torch.cat(all_raw_preds)
        all_expert_labels = torch.cat(all_expert_labels)
        all_defer_idx = torch.cat(all_defer_idx)

        return all_final_preds, all_raw_preds, all_defer_idx, all_labels, all_expert_labels