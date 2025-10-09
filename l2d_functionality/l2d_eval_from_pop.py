import torch
from tqdm import tqdm

class L2D_Eval_Population:
    
    def __init__(self, cnn, encoder, l2d, expert_pop, num_classes, device):
        self.cnn = cnn
        self.encoder = encoder
        self.l2d = l2d
        self.expert_pop = expert_pop
        self.num_classes = num_classes
        self.device = device
        
    def evaluate_single_expert(self, test_loader, expert_idx=0, psi_test=None):
        """
        Evaluate model with a single expert at deployment time.
        
        Args:
            test_loader: DataLoader for test data
            expert_idx: Which expert to use (default: 0)
            psi_test: Pre-sampled expert abilities [E, K]. If None, samples new ones.
            
        Returns:
            dict with evaluation metrics
        """
        self.cnn.eval()
        self.encoder.eval()
        self.l2d.eval()
        
        if psi_test is None:
            psi_test = self.expert_pop.sample_expert_probabilities()
        
        psi_single = psi_test[expert_idx:expert_idx+1]  # [1, K]
        phi_single = self.encoder(psi_single.to(self.device))  # [1, phi_dim]
        
        all_preds, all_labels, all_expert_preds = [], [], []
        all_classifier_preds = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Eval (expert {expert_idx})"):
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                
                # Simulate expert predictions
                expert_preds = self.expert_pop.sample_predictions_given_phi(
                    labels.cpu(), psi_single.cpu()
                )[:, 0].to(self.device)  # [N]
                
                # Forward pass
                f_x = self.cnn(imgs)  # [B, feature_dim]
                g_classes, g_perp = self.l2d(f_x, phi_single)  # [B,K], [B,1]
                
                # logits
                logits = torch.cat([g_classes, g_perp], dim=1)  # [B, K+1]
                preds = torch.argmax(logits, dim=1)  # [B]
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                all_expert_preds.append(expert_preds.cpu())
                all_classifier_preds.append(torch.argmax(g_classes, dim=1).cpu())
        
        # Concatenate all batches
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        all_expert_preds = torch.cat(all_expert_preds)
        all_classifier_preds = torch.cat(all_classifier_preds)
        
        metrics = self._compute_metrics(
            all_preds, all_labels, all_expert_preds, all_classifier_preds
        )
        metrics['psi_single'] = psi_single
        
        return metrics
    
    def _compute_metrics(self, all_preds, all_labels, all_expert_preds, all_classifier_preds):
        """evaluation metrics"""
        N = len(all_labels)
        is_class_pred = all_preds < self.num_classes
        is_expert_pred = ~is_class_pred
        
        # Classifier stats
        class_count = int(is_class_pred.sum().item())
        class_correct = int(
            (all_preds[is_class_pred] == all_labels[is_class_pred]).sum().item()
        ) if class_count > 0 else 0
        class_acc = 100.0 * class_correct / class_count if class_count > 0 else 0.0
        
        # Expert stats
        num_deferred = int(is_expert_pred.sum().item())
        expert_correct = int(
            (all_expert_preds[is_expert_pred] == all_labels[is_expert_pred]).sum().item()
        ) if num_deferred > 0 else 0
        expert_acc = 100.0 * expert_correct / num_deferred if num_deferred > 0 else 0.0
        
        # Overall accuracy
        overall_acc = 100.0 * (class_correct + expert_correct) / N
        deferral_rate = 100.0 * num_deferred / N
        
        # Pure classifier (no deferral)
        classifier_only_correct = (all_classifier_preds == all_labels).sum().item()
        classifier_only_acc = 100.0 * classifier_only_correct / N
        
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
    
    def print_results(self, metrics):
        print(f"Total test samples: {metrics['num_samples']}")
        print(f"Overall accuracy: {metrics['overall_accuracy']:.2f}%")
        print(f"Classifier accuracy (on {metrics['num_classifier']} non-deferred): {metrics['classifier_accuracy']:.2f}%")
        print(f"Classifier accuracy (if predicted all): {metrics['classifier_only_accuracy']:.2f}%")
        print(f"Expert accuracy (on {metrics['num_deferred']} deferred): {metrics['expert_accuracy']:.2f}%")
        print(f"Deferral rate: {metrics['deferral_rate']:.2f}%")
        
        if 'psi_single' in metrics:
            print(f"\nExpert abilities (psi):")
            print(metrics['psi_single'])