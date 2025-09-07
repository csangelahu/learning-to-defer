import torch
import torch.nn as nn

class LogisticRegressionModule(nn.Module):

    def __init__(self, input_dim, output_dim, deferral_class_index, l2_lambda):
        super(LogisticRegressionModule, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.l2_lambda = l2_lambda
        self.deferral_class_index = deferral_class_index

    def forward(self, x):
        x = self.linear(x)
        return x

    def l2_regularization(self):
        l2_reg = torch.tensor(0., requires_grad=True)

        # Exclude the weights connected to the deferral class
        weights_to_regularize = torch.cat([self.linear.weight[:, :self.deferral_class_index], self.linear.weight[:, self.deferral_class_index + 1:]], dim=1)
        l2_reg = torch.sum(weights_to_regularize**2)

        return self.l2_lambda * l2_reg


class OneHiddenLayerNN(nn.Module):
    def __init__(self, input_dim, output_dim, deferral_class_index, l2_lambda, hidden_dim=128):
        super(OneHiddenLayerNN, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.output = nn.Linear(hidden_dim, output_dim)

        self.l2_lambda = l2_lambda
        self.deferral_class_index = deferral_class_index

    def forward(self, x):
        x = self.hidden(x)
        x = self.activation(x)
        x = self.output(x)
        return x

    def l2_regularization(self):
        l2_reg = torch.tensor(0., device=self.output.weight.device)

        # Regularize hidden layer weights
        l2_reg += torch.sum(self.hidden.weight ** 2)

        # Regularize output layer weights but exclude deferral class
        output_weights_to_regularize = torch.cat([
            self.output.weight[:, :self.deferral_class_index],
            self.output.weight[:, self.deferral_class_index + 1:]
        ], dim=1)
        l2_reg += torch.sum(output_weights_to_regularize ** 2)

        return self.l2_lambda * l2_reg


