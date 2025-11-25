import numpy as np


class FNN:
    def __init__(self, input_seq, hidden_size, hidden_layers):
        self.input_seq = input_seq
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

    def weight_init(self):
        rows, columns = self.input_seq.shape

        self.weights = []
        self.bias = []

        for i in range(self.hidden_layers + 1):
            if i == 0:
                w = np.random.rand(columns, self.hidden_size)
                b = np.random.rand(rows, self.hidden_size)

            elif i == self.hidden_layers:
                w = np.random.rand(self.hidden_size, columns)
                b = np.random.rand(rows, columns)

            else:
                w = np.random.rand(self.hidden_size, self.hidden_size)
                b = np.random.rand(rows, self.hidden_size)

            self.weights.append(w)
            self.bias.append(b)

    def forward(self):

        z = self.input_seq

        print("====FNN======")
        for weights, bias in zip(self.weights, self.bias):
            z = z @ weights + bias

        print(z)
        self.output_fnn=z