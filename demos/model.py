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
        # l2_reg = torch.sum(self.linear.weight[:, :-1]**2)

        return self.l2_lambda * l2_reg


