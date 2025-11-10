import numpy as np
import torch
import matplotlib.pyplot as plt

class L2D_Eval_Combined:
    def __init__(self, num_classes, true_labels, predicted_labels, defer_idx):
        """
        Multi-expert evaluation without needing raw expert labels.

        Args:
            num_classes (int): Number of classes.
            true_labels (Tensor): Ground truth labels (N).
            predicted_labels (Tensor): Combined predictions with expert corrections (N).
            defer_idx (Tensor): Defer indices per sample (N), 0=classifier, 1..J=experts.
        """
        self.num_classes = num_classes
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.defer_idx = defer_idx.int()
        self.num_experts = self.defer_idx.max().item()  # max defer_idx value = number of experts

        self.evaluate()

    def evaluate(self):
        total = len(self.true_labels)

        classifier_mask = self.defer_idx == 0
        deferred_mask = self.defer_idx > 0

        correct_classifier = (self.predicted_labels == self.true_labels) & classifier_mask
        self.correct_classifier = correct_classifier.sum().item()

        correct_expert = (self.predicted_labels == self.true_labels) & deferred_mask
        self.correct_expert = correct_expert.sum().item()

        self.total_deferred = deferred_mask.sum().item()
        self.total_non_deferred = total - self.total_deferred

        self.accuracy_nondeferred = (
            self.correct_classifier / self.total_non_deferred if self.total_non_deferred > 0 else 0.0
        )
        self.deferral_accuracy = (
            self.correct_expert / self.total_deferred if self.total_deferred > 0 else 0.0
        )
        self.overall_accuracy = (self.correct_classifier + self.correct_expert) / total

        self.report = {
            'Accuracy (excluding deferrals)': self.accuracy_nondeferred,
            'Deferral Accuracy': self.deferral_accuracy,
            'Deferral Rate': self.total_deferred / total,
            'Overall Accuracy': self.overall_accuracy,
        }

        # Per-class metrics
        self.class_accuracy_classifier = []
        self.class_accuracy_expert = []
        self.class_deferral_rate = []

        for i in range(self.num_classes):
            class_mask = self.true_labels == i
            class_deferred_mask = deferred_mask & class_mask
            class_classifier_mask = classifier_mask & class_mask

            class_correct_classifier = (correct_classifier & class_mask).sum().item()
            class_correct_expert = (correct_expert & class_mask).sum().item()

            class_total = class_mask.sum().item()
            num_deferred = class_deferred_mask.sum().item()
            num_nondeferred = class_classifier_mask.sum().item()

            acc_classifier = class_correct_classifier / num_nondeferred if num_nondeferred > 0 else 0.0
            acc_expert = class_correct_expert / num_deferred if num_deferred > 0 else 0.0
            deferral_rate = num_deferred / class_total if class_total > 0 else 0.0

            self.class_accuracy_classifier.append(acc_classifier)
            self.class_accuracy_expert.append(acc_expert)
            self.class_deferral_rate.append(deferral_rate)

        # Per-expert metrics: usage and accuracy
        self.expert_metrics = {}
        for j in range(1, self.num_experts + 1):
            expert_mask = self.defer_idx == j
            total_j = expert_mask.sum().item()
            if total_j > 0:
                correct_j = (self.predicted_labels[expert_mask] == self.true_labels[expert_mask]).sum().item()
                acc_j = correct_j / total_j
                usage_percent = total_j / total
            else:
                acc_j = 0.0
                usage_percent = 0.0
            self.expert_metrics[j-1] = {
                'deferral_rate': usage_percent,
                'accuracy': acc_j,
                'total': total_j,
            }

    def model_report(self):
        print(f'Classifier Accuracy (excl. deferrals): {self.accuracy_nondeferred:.4f}')
        print(f'Expert Deferral Accuracy: {self.deferral_accuracy:.4f}')
        print(f'Deferral Rate: {self.report["Deferral Rate"]:.4f}')
        print(f'Overall Accuracy: {self.overall_accuracy:.4f}')
        return self.report

    def per_class_report(self):
        for i in range(self.num_classes):
            print(f"Class {i}: Classifier Acc: {self.class_accuracy_classifier[i]:.4f}, "
                  f"Expert Acc: {self.class_accuracy_expert[i]:.4f}, "
                  f"Deferral Rate: {self.class_deferral_rate[i]:.4f}")
        return {
            i: {
                'accuracy_classifier': self.class_accuracy_classifier[i],
                'accuracy_expert': self.class_accuracy_expert[i],
                'deferral_rate': self.class_deferral_rate[i]
            } for i in range(self.num_classes)
        }

    def expert_report(self):
        print("=== Expert Report ===")
        for j, metrics in self.expert_metrics.items():
            print(f"Expert {j}: Used {metrics['total']} times "
                  f"({metrics['deferral_rate']*100:.2f}% of samples), "
                  f"Accuracy = {metrics['accuracy']*100:.2f}%")
        return self.expert_metrics

    def visualize_metrics(self, save_path="per_class_metrics.png"):
        x = np.arange(self.num_classes)
        width = 0.25
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        rects1 = ax1.bar(x - width, self.class_accuracy_classifier, width, label='Classifier Accuracy', color='b')
        rects2 = ax1.bar(x, self.class_accuracy_expert, width, label='Expert Accuracy', color='g')
        rects3 = ax2.bar(x + width, self.class_deferral_rate, width, label='Deferral Rate', color='r')

        ax1.set_ylabel('Accuracy')
        ax2.set_ylabel('Deferral Rate')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i) for i in range(self.num_classes)])
        plt.title('Learning-to-Defer Per-Class Metrics')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        fig.tight_layout()
        plt.savefig(save_path)
        plt.show()
