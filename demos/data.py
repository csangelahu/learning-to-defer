import torch
from torch.utils.data import Dataset, DataLoader
import random

class ExpertCombinedDataset(Dataset):
    def __init__(self, dataset, expert_labels):
        self.dataset = dataset
        self.expert_labels = expert_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        expert_label = self.expert_labels[idx]
        return image, label, expert_label

class ExpertMNIST:  
    def __init__(self, base_prob):
        self.base_prob = base_prob

    def expert_prediction(self, true_label):
        if self.base_prob == 1: 
            return true_label 

        assert 0 <= true_label < 10, "true label must be a single digit"
        if true_label % 2 == 0:
            return (true_label + 1) % 10 # always incorrect 
        else:
            # return true_label # for testing always correct on 50% of cases 
            bernoulli_prob = self.base_prob + true_label / 100
            if random.random() < bernoulli_prob:  # Sample from Bernoulli
                return true_label  # Correct prediction
            else:
                return (true_label + 1) % 10   # Incorrect prediction
            
class ExpertCIFAR:
    def __init__(self, k):
        assert 1 <= k <= 10, "k must be an integer in the range [1, 10]"
        self.k = k

    def expert_prediction(self, true_label):
        assert 0 <= true_label < 10, "true label must be a single digit in the range [0, 9]"
        
        if true_label < self.k:
            return true_label  # Expert predicts correctly for the first k classes
        else:
            return random.randint(0, 9)  # Expert predicts randomly from all classes


def get_dataloaders(train_dataset_raw, expert_labels_train, test_dataset_raw, expert_labels_test, batch_size=64):

    train_dataset = ExpertCombinedDataset(train_dataset_raw, expert_labels_train)
    test_dataset = ExpertCombinedDataset(test_dataset_raw, expert_labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader 