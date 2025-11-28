import numpy as np

class FNN:
    def __init__(self, input_seq, hidden_size, hidden_layers):
        self.input_seq = input_seq
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers

    def handel_activation(matrix):
        new_matrix=[]
        for row in matrix:
            new_row=[]
            for value in row:
                if value>0:
                    new_row.append(value)

                else:
                    new_row.append(0)

            new_matrix.append(new_row)

        return np.array(new_matrix)

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

    def forward(self,input_from_last_layer):

        self.input_from_last_layer=input_from_last_layer
        self.f_pass_values=[]

        z = self.input_from_last_layer

        idx=0
        for weights, bias in zip(self.weights, self.bias):
            layer_out=[]

            z = z @ weights + bias
            print("z @ w + b",z,end="\n\n")
            layer_out.append(z)

            if idx < len(self.weights) - 1:
                z = FNN.handel_activation(z)
                print("ReLU",z,end="\n\n")
                layer_out.append(z)

            self.f_pass_values.append(layer_out)
            idx+=1

    def backpropagation(self,gradient_from_last_layer):
        
        print("============== bp ===============\n")
        f_pass=self.f_pass_values[::-1]
        weights=self.weights[::-1]

        gradient=gradient_from_last_layer

        # =========================================================
        # delta = f_pass[1][0].T @ gradient
        # print(delta,end="\n\n")

        # =========================================================        
        # gradient = gradient @ weights[0].T * f_pass[1][0]

        # delta = gradient.T @ f_pass[2][0]
        # print(delta,end="\n\n")

        # # =========================================================
        # gradient = gradient @ weights[1].T * f_pass[2][0]

        # delta = self.input_from_last_layer.T @ gradient 
        # print(delta,end="\n\n")

        for idx in range(len(f_pass)):
            if idx==0:
                delta = f_pass[idx+1][0].T @ gradient  
                print(delta,end="\n\n")

            elif idx==len(f_pass)-1:
                gradient= (gradient @ weights[idx-1].T )* f_pass[idx][0]
                delta = self.input_from_last_layer.T @ gradient 
                print(delta,end="\n\n")


            else:
                gradient= (gradient @ weights[idx-1].T ) * f_pass[idx][0]

                delta =gradient.T @ f_pass[idx+1][0] 
                print(delta,end="\n\n")
        
        gradient_to_next_layer=gradient @ weights[-1].T
        print("->>>>>>",gradient_to_next_layer)