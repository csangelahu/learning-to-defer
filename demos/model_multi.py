import torch
import torch.nn as nn

class LogisticRegressionMultiExpert(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts, l2_lambda):
        super(LogisticRegressionMultiExpert, self).__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.output_dim = num_classes + num_experts

        self.linear = nn.Linear(input_dim, self.output_dim)
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def l2_regularization(self):
        """
        L2 regularization excluding expert logits (the last num_experts columns).
        """
        if self.l2_lambda == 0:
            return torch.tensor(0.0, device=self.linear.weight.device)

        # Regularize only class-related weights
        weights_to_regularize = self.linear.weight[:, :self.num_classes]
        l2_reg = torch.sum(weights_to_regularize ** 2)
        return self.l2_lambda * l2_reg


class OneHiddenLayerNNMultiExpert(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts, l2_lambda, hidden_dim=128):
        super(OneHiddenLayerNNMultiExpert, self).__init__()
        self.num_classes = num_classes
        self.num_experts = num_experts
        self.output_dim = num_classes + num_experts

        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, self.output_dim)

        self.l2_lambda = l2_lambda

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

    def l2_regularization(self):
        if self.l2_lambda == 0:
            return torch.tensor(0.0, device=self.output.weight.device)

        l2_reg = torch.sum(self.hidden.weight ** 2)

        # Only regularize class weights (exclude expert columns)
        class_weights = self.output.weight[:, :self.num_classes]
        l2_reg += torch.sum(class_weights ** 2)

        return self.l2_lambda * l2_reg


class CustomMLPMultiExpert(nn.Module):
    def __init__(self, input_dim, num_classes, num_experts, l2_lambda,
                 hidden_dims=[512, 256, 128, 128], activation=nn.ReLU, dropout_rate=0.0):
        super(CustomMLPMultiExpert, self).__init__()

        self.num_classes = num_classes
        self.num_experts = num_experts
        self.output_dim = num_classes + num_experts

        self.layers = nn.ModuleList()
        self.activation_fn = activation()
        self.dropout = nn.Dropout(dropout_rate)

        # Hidden layers
        in_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(in_dim, h_dim))
            in_dim = h_dim

        # Output layer
        self.output = nn.Linear(in_dim, self.output_dim)

        self.l2_lambda = l2_lambda

    def forward(self, x):
        for layer in self.layers:
            x = self.activation_fn(layer(x))
            x = self.dropout(x)
        x = self.output(x)
        return x

    def l2_regularization(self):
        if self.l2_lambda == 0:
            return torch.tensor(0.0, device=self.output.weight.device)

        l2_reg = torch.tensor(0.0, device=self.output.weight.device)

        # L2 on all hidden layer weights
        for layer in self.layers:
            l2_reg += torch.sum(layer.weight ** 2)

        # Only regularize output weights for class logits (not expert logits)
        class_weights = self.output.weight[:, :self.num_classes]
        l2_reg += torch.sum(class_weights ** 2)

        return self.l2_lambda * l2_reg
