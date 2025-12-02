import numpy as np

class FNN:
    def __init__(self, input_seq, hidden_size, hidden_layers,verbose=False):
        self.input_seq = input_seq
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.verbose = verbose

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
    
    def derivative_of_ReLU(matrix):
        new_matrix=[]
        for row in matrix:
            new_row=[]
            for value in row:
                if value > 0:
                    new_row.append(1)

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
        if self.verbose:
            print("------------- weights ------------------")
            idx=0
            for w,b in zip(self.weights,self.bias):
                if len(self.weights)-1 != idx:
                    print(f"w{idx} -> {w}")
                    print(f"b{idx} -> {b}",end="\n\n")
                else:
                    print(f"wo -> {w}")
                    print(f"bo -> {b}",end="\n\n")
                idx+=1
            print("------------- forwarpass ------------------")

        self.input_from_last_layer=input_from_last_layer
        self.f_pass_values=[]

        z = self.input_from_last_layer

        idx=0
        for weights, bias in zip(self.weights, self.bias):
            layer_out=[]

            z = z @ weights + bias

            if self.verbose:
                print("z @ w + b -> ",z,end="\n\n")
            
            layer_out.append(z)

            if idx < len(self.weights) - 1:
                z = FNN.handel_activation(z)
                
                if self.verbose:
                    print("ReLU",z,end="\n\n")

                layer_out.append(z)

            self.f_pass_values.append(layer_out)
            idx+=1

    def backpropagation(self,gradient_from_last_layer):
        
        self.weight_and_bias_gradients=[]
        
        if self.verbose:
            print("----------------- bp -----------------\n")
        
        f_pass=self.f_pass_values[::-1]
        weights=self.weights[::-1]

        gradient=gradient_from_last_layer

        for idx in range(len(f_pass)):
            if idx==0:
                delta = f_pass[idx+1][0].T @ gradient  

            elif idx==len(f_pass)-1:
                gradient= (gradient @ weights[idx-1].T )* FNN.derivative_of_ReLU(f_pass[idx][0])
                delta = self.input_from_last_layer.T @ gradient 

            else:
                gradient= (gradient @ weights[idx-1].T ) * FNN.derivative_of_ReLU(f_pass[idx][0])

                delta =gradient.T @ f_pass[idx+1][0] 

            if self.verbose:
                print("gradient for bias ->",gradient)
                print("gradient for weight ->",delta,end="\n\n")

            self.weight_and_bias_gradients.append([gradient,delta])
        
        self.gradient_to_next_layer=gradient @ weights[-1].T
        if self.verbose:
            print("gradient to next layer -> ",self.gradient_to_next_layer)

    def update_weights(self,lr):
        if self.verbose:
            print("----------- updating weights --------------")
        
        idx=0
        for update,weight,bias, in zip(self.weight_and_bias_gradients[::-1],self.weights,self.bias):
            bias -= update[0] * lr 
            weight -= update[1] * lr
            if self.verbose:
                if idx != len(self.weights)-1:
                    print(f"w{idx} -> ",weight,end="\n\n")
                    print(f"b{idx} ->" ,bias,end="\n\n")
                else:
                    print(f"wo -> ",weight,end="\n\n")
                    print(f"bo ->" ,bias,end="\n\n")
            idx+=1