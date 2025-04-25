import torch
import numpy as np
import matplotlib.pyplot as plt

class L2D_Eval:
    def __init__(self, num_classes, outputs, labels, expert_labels, deferral_class_index, realizable_sm=False, tau=0.5):
        """
        Initializes the L2D_Eval class for evaluating a Learning-to-Defer model.

        Args:
            num_classes (int): Number of target classes (excluding deferral).
            outputs (Tensor): Model logits of shape [batch_size, num_classes + 1].
            labels (Tensor): Ground-truth labels.
            expert_labels (Tensor): Expert-provided labels used for deferral.
            deferral_class_index (int): Index corresponding to the deferral class in outputs.
            realizable_sm (bool, optional): Whether to use a realizable softmax strategy for deferral. Default is False.
            tau (float, optional): Threshold for the deferral decision in realizable softmax.
        """

        self.num_classes = num_classes
        self.outputs = outputs
        self.labels = labels
        self.expert_labels = expert_labels
        self.deferral_class_index = deferral_class_index
        self.realizable_sm = realizable_sm
        self.tau = tau
        self.class_metrics = {}

        self.evaluate_model()


    def model_report(self):
        """
        Prints and returns the overall performance metrics of the model including:
        - Accuracy of the pure classifier (no deferrals)
        - Accuracy excluding deferred samples
        - Accuracy of deferred samples handled by the expert
        - Deferral rate
        - Combined (overall) accuracy considering both classifier and expert

        Returns:
            dict: Summary report with the described metrics.
        """

        print(f'Test Accuracy of Regular Classifier: {self.accuracy_pure_classifier:.2f}%')
        print(f'Test Accuracy (excluding deferrals): {self.accuracy_nondeferred:.2f}%')
        print(f'Deferral Accuracy: {self.deferral_accuracy:.2f}%')
        print(f'Overall Deferral Rate: {self.deferral_rate:.2f}%')
        print(f'Overall L2D Accuracy: {self.overall_accuracy:.2f}%')

        return self.report


    def per_class_report(self):
        """
        Prints and returns per-class metrics including:
        - Accuracy excluding deferred samples
        - Deferral rate
        - Overall accuracy (classifier + expert)

        Returns:
            dict: Per-class performance metrics.
        """

        for i in range(self.num_classes):
            metrics = self.class_metrics.get(i, {})
            print(f"Class {i}: Accuracy (non-deferred) = {metrics.get('accuracy_nondeferred', 0.0):.2f}%, "
                  f"Deferral Rate = {metrics.get('deferral_rate', 0.0):.2f}%, "
                  f"Overall Accuracy = {metrics.get('overall_accuracy', 0.0):.2f}%")
            
        return self.class_metrics


    def rejector(self): # for realizable_sm = True 
        """
        Computes a binary mask indicating which samples should be deferred 
        based on the realizable softmax decision rule.
        Used only if `realizable_sm=True`.

        Returns:
            Tensor (bool): Mask where True indicates a deferred sample to the expert.
        """

        g_perp = self.outputs[:, self.deferral_class_index]
        
        outputs_without_deferral = self.outputs.clone()
        outputs_without_deferral[:, self.deferral_class_index] = -float('inf')  # Mask the deferral class
        
        max_other_classes = outputs_without_deferral.max(dim=1)[0]
        
        # Reject if g_perp(x) - max_y g_y(x) >= tau
        reject_mask = (g_perp - max_other_classes) >= self.tau
        return reject_mask
    
        # if r(x) = 1, deferred to human; if r(x) = 0, classifier makes final decision 
    
    def evaluate_model(self):

        """
        Performs a full evaluation of the model, and updates internal attributes.
        """

        with torch.no_grad():
            correct = 0
            correct_pure_classifier = 0
            total = 0
            deferred_correct = 0
            total_deferred = 0

            # Store results for each class
            class_correct_nondeferred = {i: 0 for i in range(self.num_classes)}
            class_total = {i: 0 for i in range(self.num_classes)}
            class_deferred = {i: 0 for i in range(self.num_classes)}
            class_correct_overall = {i: 0 for i in range(self.num_classes)}  # Overall correct per class


            outputs_pure_classifier = self.outputs.clone()
            outputs_pure_classifier[:, self.deferral_class_index] = -float('inf')

            

            _, predicted_pure_classifier = torch.max(outputs_pure_classifier.data, 1)

            if self.realizable_sm:
                is_deferred = self.rejector()  # Boolean mask
                predicted = predicted_pure_classifier.clone()
                predicted[is_deferred] = self.deferral_class_index
            else:
                _, predicted = torch.max(self.outputs.data, 1)
                is_deferred = predicted == self.deferral_class_index



            is_correct = predicted == self.labels  # Correct non-deferred 
            is_correct_pure_classifier = predicted_pure_classifier == self.labels 

            # Evaluate correctly classified samples from non-deferred
            correct += (is_correct & ~is_deferred).sum().item()
            total += (~is_deferred).sum().item()

            # Evaluate correct where model classified everything 
            correct_pure_classifier += is_correct_pure_classifier.sum().item()

            # Evaluate correct deferrals
            expert_predictions = self.expert_labels[is_deferred]
            correct_deferrals = expert_predictions == self.labels[is_deferred]
            deferred_correct += correct_deferrals.sum().item()
            total_deferred += is_deferred.sum().item()

            # results for each class class
            for i in range(self.num_classes):  
                class_mask = self.labels == i
                class_correct_nondeferred[i] += (is_correct & ~is_deferred & class_mask).sum().item()
                class_total[i] += class_mask.sum().item()
                class_deferred[i] += (is_deferred & class_mask).sum().item()

                deferred_class_indices = torch.where(is_deferred & class_mask)[0]  # Indices of deferred samples for this class
                if len(deferred_class_indices) > 0:
                    is_deferred_indices = torch.where(is_deferred)[0]
                    
                    # Identify the indices of the deferred_class_indices in is_deferred_indices
                    deferred_mask_in_is_deferred = torch.isin(is_deferred_indices, deferred_class_indices)
                    
                    class_deferred_correct = correct_deferrals[deferred_mask_in_is_deferred]
                    class_correct_overall[i] += (is_correct & ~is_deferred & class_mask).sum().item() + class_deferred_correct.sum().item()
                else:
                    class_correct_overall[i] += (is_correct & ~is_deferred & class_mask).sum().item()


            # accuracy if the model had classified everything
            self.accuracy_pure_classifier = 100 * correct_pure_classifier / len(self.labels)
            self.accuracy_nondeferred = 100 * correct / total if total > 0 else 0.0
            self.deferral_accuracy = 100 * deferred_correct / total_deferred if total_deferred > 0 else 0.0
            self.deferral_rate = 100 * total_deferred / len(self.labels)  # Overall deferral rate
            self.overall_accuracy = 100 * (correct + deferred_correct) / len(self.labels)

            self.report = {
                'Test Accuracy of Regular Classifier': self.accuracy_pure_classifier,
                'Test Accuracy (excluding deferrals)': self.accuracy_nondeferred,
                'Deferral Accuracy': self.deferral_accuracy,
                'Overall Deferral Rate': self.deferral_rate,
                'Overall L2D Accuracy': self.overall_accuracy
            }

            # Print per-class accuracy and deferral rate
            self.class_accuracy_list = []
            self.class_deferral_rate_list = []
            self.class_overall_accuracy_list = []

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

            return
        
            
    
    def visualize_per_class_metrics(self, save_path="per_class_metrics.png"):

        """
        Plots a bar chart showing:
        - Non-deferred accuracy
        - Overall accuracy (including expert)
        - Deferral rate

        Args:
        save_path (str, optional): Path where the plot image will be saved. 
                                    Default is 'per_class_metrics.png'.

        Saved as 'per_class_metrics.png'.
        """

        x = np.arange(self.num_classes)  
        width = 0.3
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        rects1 = ax1.bar(x - width, self.class_accuracy_list, width, label='Non-deferred Accuracy', color='b')
        rects2 = ax1.bar(x, self.class_overall_accuracy_list, width, label='Overall Accuracy', color='g')
        rects3 = ax2.bar(x + width, self.class_deferral_rate_list, width, label='Deferral Rate', color='r')

        ax1.set_ylabel('Accuracy (%)', color='b')
        ax2.set_ylabel('Deferral Rate (%)', color='r')
        ax1.set_xticks(x)
        ax1.set_xticklabels([str(i) for i in range(10)])
        plt.title('Accuracy and Deferral Rate per Class')

        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.savefig(save_path)