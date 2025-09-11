from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F




class L2D_Eval_Multi_Expert:
    def __init__(self, num_classes, outputs, labels, expert_labels):
        """
        Initializes the evaluation for a multi-expert Learning-to-Defer model.

        Args:
            num_classes (int): Number of target classes (K).
            outputs (Tensor): Model logits of shape [batch_size, K + J].
            labels (Tensor): Ground-truth labels of shape [batch_size].
            expert_labels (Tensor): Expert predictions of shape [batch_size, J].
        """
        self.num_classes = num_classes
        self.num_experts = expert_labels.size(1) if expert_labels is not None else 0
        self.outputs = outputs
        self.labels = labels
        self.expert_labels = expert_labels
        self.class_metrics = {}
        self.evaluate_model()

    def rejector_multi_expert(self, logits, num_classes):
        """
        Determines the final prediction based on the multi-expert rejection rule.

        Args:
            logits (Tensor): Model output logits of shape (batch_size, K + J),
                            where K is num_classes and J is num_experts.
            num_classes (int): The number of classes (K).

        Returns:
            Tensor: A tensor of final predictions (batch_size,). Values will be
                    class indices (0 to K-1) or deferral indices (K to K+J-1).
        """
        class_logits = logits[:, :num_classes]
        expert_logits = logits[:, num_classes:]

        class_scores, predicted_classes = torch.max(class_logits, dim=1)

        # best expert to defer to
        expert_scores, predicted_experts_relative = torch.max(expert_logits, dim=1)
        
        # Convert to the global logit index
        num_experts = expert_logits.size(1)
        if num_experts > 0:
            predicted_experts_global = predicted_experts_relative + num_classes
        else: # Handle case with no experts
            predicted_experts_global = torch.tensor([])


        # Determine which samples to defer
        if num_experts > 0:
            defer_mask = (class_scores <= expert_scores)
        else:
            defer_mask = torch.zeros_like(class_scores, dtype=torch.bool)

        final_predictions = predicted_classes.clone()
        if num_experts > 0:
            final_predictions[defer_mask] = predicted_experts_global[defer_mask]

        return final_predictions


    def evaluate_model(self):
        """
        Performs a full evaluation of the model with multi-expert deferral,
        and updates internal attributes (including per-expert stats).
        """

        with torch.no_grad():
            correct = 0
            correct_pure_classifier = 0
            total = 0
            deferred_correct = 0
            total_deferred = 0

            # Per-class stats
            class_correct_nondeferred = {i: 0 for i in range(self.num_classes)}
            class_total = {i: 0 for i in range(self.num_classes)}
            class_deferred = {i: 0 for i in range(self.num_classes)}
            class_correct_overall = {i: 0 for i in range(self.num_classes)}

            # Per-expert stats
            self.num_experts = self.outputs.size(1) - self.num_classes
            expert_total = {j: 0 for j in range(self.num_experts)}
            expert_correct = {j: 0 for j in range(self.num_experts)}

            # Pure classifier
            outputs_pure_classifier = self.outputs[:, :self.num_classes]
            _, predicted_pure_classifier = torch.max(outputs_pure_classifier.data, 1)

            predicted = self.rejector_multi_expert(self.outputs, self.num_classes)

            # Identify deferred predictions
            is_deferred = predicted >= self.num_classes
            is_correct_pure_classifier = predicted_pure_classifier == self.labels
            is_correct_nondeferred = (predicted == self.labels) & ~is_deferred

            # Non-deferred stats
            correct += is_correct_nondeferred.sum().item()
            total += (~is_deferred).sum().item()

            # Pure classifier accuracy
            correct_pure_classifier += is_correct_pure_classifier.sum().item()

            # Deferral accuracy + expert tracking
            if is_deferred.any():
                deferred_expert_indices = predicted[is_deferred] - self.num_classes
                expert_predictions = self.expert_labels[is_deferred, deferred_expert_indices]
                correct_deferrals = expert_predictions == self.labels[is_deferred]

                deferred_correct += correct_deferrals.sum().item()
                total_deferred += is_deferred.sum().item()

                # Per-expert counts
                for j in range(self.num_experts):
                    expert_mask = deferred_expert_indices == j
                    expert_total[j] += expert_mask.sum().item()
                    if expert_mask.any():
                        expert_correct[j] += (expert_predictions[expert_mask] == self.labels[is_deferred][expert_mask]).sum().item()

            # Per-class stats
            for i in range(self.num_classes):
                class_mask = self.labels == i

                # Non-deferred correct
                class_correct_nondeferred[i] += (is_correct_nondeferred & class_mask).sum().item()

                # Totals
                class_total[i] += class_mask.sum().item()
                class_deferred[i] += (is_deferred & class_mask).sum().item()

                # Overall correct
                deferred_mask = is_deferred & class_mask
                if deferred_mask.any():
                    deferred_expert_indices = predicted[deferred_mask] - self.num_classes
                    class_expert_predictions = self.expert_labels[deferred_mask, deferred_expert_indices]
                    class_deferred_correct = (class_expert_predictions == self.labels[deferred_mask]).sum().item()
                else:
                    class_deferred_correct = 0

                class_correct_overall[i] += (is_correct_nondeferred & class_mask).sum().item() \
                                            + class_deferred_correct

            # overall metrics
            self.accuracy_pure_classifier = 100 * correct_pure_classifier / len(self.labels)
            self.accuracy_nondeferred = 100 * correct / total if total > 0 else 0.0
            self.deferral_accuracy = 100 * deferred_correct / total_deferred if total_deferred > 0 else 0.0
            self.deferral_rate = 100 * total_deferred / len(self.labels)
            self.overall_accuracy = 100 * (correct + deferred_correct) / len(self.labels)

            self.report = {
                'Test Accuracy of Regular Classifier': self.accuracy_pure_classifier,
                'Test Accuracy (excluding deferrals)': self.accuracy_nondeferred,
                'Deferral Accuracy': self.deferral_accuracy,
                'Overall Deferral Rate': self.deferral_rate,
                'Overall L2D Accuracy': self.overall_accuracy
            }

            # Per-class metrics
            self.class_accuracy_list = []
            self.class_deferral_rate_list = []
            self.class_overall_accuracy_list = []
            self.class_metrics = {}

            for i in range(self.num_classes):
                class_accuracy_nondeferred = 100 * class_correct_nondeferred[i] / (class_total[i] - class_deferred[i]) if (class_total[i] - class_deferred[i]) > 0 else 0.0
                class_deferral_rate = 100 * class_deferred[i] / class_total[i] if class_total[i] > 0 else 0.0
                class_overall_accuracy = 100 * class_correct_overall[i] / class_total[i] if class_total[i] > 0 else 0.0

                self.class_accuracy_list.append(class_accuracy_nondeferred)
                self.class_deferral_rate_list.append(class_deferral_rate)
                self.class_overall_accuracy_list.append(class_overall_accuracy)

                self.class_metrics[i] = {
                    'accuracy_nondeferred': class_accuracy_nondeferred,
                    'deferral_rate': class_deferral_rate,
                    'overall_accuracy': class_overall_accuracy
                }

            # Per-expert metrics
            self.expert_metrics = {}
            for j in range(self.num_experts):
                acc = 100 * expert_correct[j] / expert_total[j] if expert_total[j] > 0 else 0.0
                rate = 100 * expert_total[j] / len(self.labels)
                self.expert_metrics[j] = {
                    'deferral_rate': rate,
                    'accuracy': acc,
                    'total': expert_total[j]
                }

            return


    def model_report(self):

        print(f'Test Accuracy of Pure Classifier: {self.accuracy_pure_classifier:.2f}%')
        print(f'Test Accuracy (excluding deferrals): {self.accuracy_nondeferred:.2f}%')
        print(f'Deferral Accuracy (Experts): {self.deferral_accuracy:.2f}%')
        print(f'Overall Deferral Rate: {self.deferral_rate:.2f}%')
        print(f'Overall Combined Accuracy: {self.overall_accuracy:.2f}%')
        return self.report
    
    def per_class_report(self):

        for i in range(self.num_classes):
            metrics = self.class_metrics.get(i, {})
            print(f"Class {i}: Accuracy (non-deferred) = {metrics.get('accuracy_nondeferred', 0.0):.2f}%, "
                  f"Deferral Rate = {metrics.get('deferral_rate', 0.0):.2f}%, "
                  f"Overall Accuracy = {metrics.get('overall_accuracy', 0.0):.2f}%")
            
        return self.class_metrics
    
    def expert_report(self):
        """
        Prints and returns per-expert performance metrics.
        """
        for j, metrics in self.expert_metrics.items():
            print(f"Expert {j}: Used {metrics['total']} times "
                f"({metrics['deferral_rate']:.2f}% of samples), "
                f"Accuracy = {metrics['accuracy']:.2f}%")
        return self.expert_metrics

    def visualize_per_class_metrics(self, save_path="per_class_metrics.png"):
        """
        Plots a bar chart showing:
        - Non-deferred accuracy
        - Overall accuracy (including expert deferrals)
        - Deferral rate

        Args:
            save_path (str, optional): Path where the plot image will be saved. 
        """

        x = np.arange(self.num_classes)  
        width = 0.3
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        rects1 = ax1.bar(x - width, self.class_accuracy_list, width,
                        label='Non-deferred Accuracy', color='b')
        rects2 = ax1.bar(x, self.class_overall_accuracy_list, width,
                        label='Overall Accuracy (incl. experts)', color='g')
        rects3 = ax2.bar(x + width, self.class_deferral_rate_list, width,
                        label='Deferral Rate', color='r')

        ax1.set_ylabel('Accuracy (%)', color='b')
        ax2.set_ylabel('Deferral Rate (%)', color='r')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i) for i in range(self.num_classes)])
        plt.title('Per-Class Accuracy and Deferral Rate (Multi-Expert L2D)')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig(save_path)


