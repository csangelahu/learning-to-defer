import torch # type: ignore
from tqdm import tqdm

class L2D_Eval_Population:
    
    def __init__(self, cnn, l2d, expert_pop, num_classes, device, expert_aware=True):
        """l2d is the expert decision module model. It should output the 
        logits for the model-predicted class labels and deferral decisions."""
        
        self.cnn = cnn
        self.l2d = l2d
        self.expert_pop = expert_pop
        self.num_classes = num_classes
        self.device = device
        self.expert_aware = expert_aware  # True for Eq.7, False for Eq.8
    
    def evaluate_single_expert(self, test_loader, expert_idx=0, psi_test=None):
        """Evaluate model with one expert active at deployment time."""
        if self.cnn is not None:
            self.cnn.eval()
        self.l2d.eval()

        if psi_test is None:
            psi_test = self.expert_pop.sample_expert_probabilities()
        
        all_preds, all_labels, all_expert_preds = [], [], []
        all_classifier_preds = []

        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Eval (expert {expert_idx})"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                # Get expert predictions for this specific expert
                expert_preds = self.expert_pop.sample_predictions_given_phi(
                    labels.cpu(),
                    psi_test[expert_idx:expert_idx+1].cpu()
                )[:, 0].to(self.device)

                if self.expert_aware:
                    # Eq.7 (BLUE) - Expert-aware
                    f_x = self.cnn(imgs)
                    g_classes, g_perp_all = self.l2d(f_x, psi_test.to(self.device))
                    
                    # Select the rejector output for the current expert
                    g_perp = g_perp_all[:, expert_idx]  # [N]
                    
                    # Create (K+1)-dimensional logits for this specific expert
                    logits = torch.cat([g_classes, g_perp.unsqueeze(1)], dim=1)  # [N, K+1]
                else:
                    # Eq.8 (RED) - Population-averaged
                    g_classes, g_perp = self.l2d(imgs)
                    logits = torch.cat([g_classes, g_perp.unsqueeze(1)], dim=1)  # [N, K+1]

                # Predictions: 0 to K-1 are classes, K means defer
                preds = torch.argmax(logits, dim=1)  # [N]

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_expert_preds.append(expert_preds.cpu())
                all_classifier_preds.append(torch.argmax(g_classes, dim=1).cpu())

        # Aggregate metrics
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_expert_preds = torch.cat(all_expert_preds)
        all_classifier_preds = torch.cat(all_classifier_preds)

        metrics = self._compute_metrics(all_preds, all_labels, all_expert_preds, all_classifier_preds)
        metrics["psi_single"] = psi_test
        return metrics

    def _compute_metrics(self, all_preds, all_labels, all_expert_preds, all_classifier_preds):
        N = len(all_labels)
        is_class_pred = all_preds < self.num_classes
        is_expert_pred = ~is_class_pred

        class_count = is_class_pred.sum().item()
        num_deferred = is_expert_pred.sum().item()
        class_correct = (all_preds[is_class_pred] == all_labels[is_class_pred]).sum().item() if class_count > 0 else 0
        expert_correct = (all_expert_preds[is_expert_pred] == all_labels[is_expert_pred]).sum().item() if num_deferred > 0 else 0

        class_acc = 100 * class_correct / class_count if class_count > 0 else 0
        expert_acc = 100 * expert_correct / num_deferred if num_deferred > 0 else 0
        overall_acc = 100 * (class_correct + expert_correct) / N
        deferral_rate = 100 * num_deferred / N
        classifier_only_acc = 100 * (all_classifier_preds == all_labels).sum().item() / N

        return {
            'overall_accuracy': overall_acc,
            'classifier_accuracy': class_acc,
            'classifier_only_accuracy': classifier_only_acc,
            'expert_accuracy': expert_acc,
            'deferral_rate': deferral_rate,
            'num_samples': N,
            'num_deferred': num_deferred,
            'num_classifier': class_count,
        }
