from attention.attention_block import AttentionBlock
from LayerNorm.ln import Norm
from fnn.fnn_block import FNN
import numpy as np
from Utils.display_weights import show
# input_seq = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# input_seq = [[1, 1], [1, 1]]
# input_seq = [[1.11, 1.45, 0.99], [0.99, 1.32, 1.44], [0.88, 0.94, 1.22]]

# input_seq = [[1.11, 1.45, 0.99,1.00], [0.99, 1.32, 1.44,1.1], [0.88, 0.94, 1.22,.77]]
# output_seq=[[1,2],[2,3],[3,4]]

# input_seq = [[0,0], [1,1], [1,0],[0,1]]
# output_seq=[[1],[1],[0],[0]]

input_seq = [[1,2,3], [5,9,8], [10,10,10], [11,1,12],[5,4,3],[9,8,9],[9,1,1],[1,3,3],[2,0,4],[5,1,9],[0,5,10],[3,13,12],[4,5,7],[1,1,1]]
output_seq = [[3,2,1], [8,9,5], [10,10,10], [12,1,11],[3,4,5],[9,8,8],[1,1,9],[3,3,1],[4,0,2],[9,1,5],[10,5,0],[12,13,3],[7,5,4],[1,1,1]]


class TransformerBlock:
    def __init__(self, input_seq, output_seq,num_transformer_layers, num_attention_heads,fnn_hidden_size):

        self.layers = []
        self.input_seq=np.array(input_seq)
        self.output_seq=np.array(output_seq)

        # print(self.input_seq,self.input_seq.shape)
        # print(self.output_seq,self.output_seq.shape)

        self.t_output_weight=np.random.rand(self.input_seq.shape[1],self.output_seq.shape[1])

        for _ in range(num_transformer_layers):


            attention_block = AttentionBlock(
                num_heads=num_attention_heads,
                input_seq=self.input_seq,
            )
            attention_block.weight_init()

            ln_1 = Norm(self.input_seq)
            ln_1.weight_init()

            fnn=FNN(self.input_seq,hidden_size=fnn_hidden_size,hidden_layers=1)
            fnn.weight_init()

            ln_2 = Norm(self.input_seq)
            ln_2.weight_init()
            
            self.layers.append([attention_block,ln_1,fnn,ln_2])

        # show(self.layers)

    # def forward_pass(self,timestep_input,timestep_output):
    def forward_pass(self):

        
        self.output_from_last_layer=self.input_seq
        # self.output_from_last_layer=timestep_input
        self.forward_pass_values=[]


        for layer in self.layers:
            attention : AttentionBlock=layer[0]
            ln_1 : Norm=layer[1]
            fnn : FNN=layer[2]
            ln_2 : Norm=layer[3]


            # attention heads
            attention.forward_pass(input_from_last_layer=self.output_from_last_layer)

            
            # add + Norm
            residual_connection_1=self.output_from_last_layer+attention.output_MHA
            ln_1.forward(residual_connection_1)
            ln_1.scale()


            # FNN
            fnn.forward(input_from_last_layer=ln_1.scaled_matrix)

            # add + Norm
            residual_connection_2=ln_1.scaled_matrix+fnn.f_pass_values[-1][0]
            ln_2.forward(residual_connection_2)
            ln_2.scale()

            self.output_from_last_layer=ln_2.scaled_matrix

            self.forward_pass_values.append(
                [
                    attention.output_MHA,
                    residual_connection_1,
                    ln_1.normalized_matrix,
                    ln_1.scaled_matrix,
                    fnn.f_pass_values[-1],
                    residual_connection_2,
                    ln_2.normalized_matrix,
                    ln_2.scaled_matrix,
                ]
            )
        self.t_output = self.output_from_last_layer @ self.t_output_weight
        self.loss = self.t_output - self.output_seq
        # self.loss = self.t_output - timestep_output

        self.error = 0.5 * np.sum(self.loss ** 2)

    def backpropagation(self,lr):
        delta_t_out_weight= self.output_from_last_layer.T @ self.loss
        self.t_output_weight -= delta_t_out_weight * lr
        
        gradient=self.loss @ self.t_output_weight.T

        for layer in self.layers[::-1]:
            ln_2:Norm = layer[-1]
            fnn:FNN = layer[-2]
            ln_1:Norm = layer[-3]
            attention_block:AttentionBlock = layer[-4]

            # layer norm
            ln_2.create_jacobian()
            ln_2.backpropagation(gradient_from_last_layer=gradient,jacobian_matrix=ln_2.jacobian_matrix)
            ln_2.update_weights(lr=lr)

            # res 1
            res_gradient_1=ln_2.gradient_to_next_layer.copy()

            # fnn
            fnn.backpropagation(gradient_from_last_layer=ln_2.gradient_to_next_layer)
            fnn.update_weights(lr=lr)

            # layer norm
            ln_1.create_jacobian()
            ln_1.backpropagation(gradient_from_last_layer=fnn.gradient_to_next_layer+res_gradient_1,jacobian_matrix=ln_1.jacobian_matrix)
            ln_1.update_weights(lr=lr)

            # res 2
            res_gradient_2=ln_1.gradient_to_next_layer.copy()

            # attention
            attention_block.backpropagation(gradient_from_last_layer=ln_1.gradient_to_next_layer)
            attention_block.update_weights(lr=lr)


            gradient=attention_block.gradient_to_next_layer + res_gradient_2


nn=TransformerBlock(input_seq=input_seq,output_seq=output_seq,num_attention_heads=3,num_transformer_layers=10,fnn_hidden_size=10)

EPOCHS=100

for _ in range(EPOCHS):
    nn.forward_pass()
    # print(f"prediction -> {np.array(nn.t_output).T} error -> {nn.error}")
    print("prediction -> ",nn.t_output,end="\n\n")
    print("error -> ",nn.error,end="\n\n")
    nn.backpropagation(lr=0.01)


    # print("\n\n==================================\n\n")
    # for idx in range(len(input_seq)):
    #     nn.forward_pass(input_seq[idx],output_seq[idx])
    #     print(f"prediction -> {np.array(nn.t_output).T} error -> {nn.error}")
    #     # print("prediction -> ",nn.t_output,end="\n\n")
    #     # print("error -> ",nn.error,end="\n\n")
    #     nn.backpropagation(lr=0.01)


