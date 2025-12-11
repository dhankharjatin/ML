from attention.attention_block import AttentionBlock
from LayerNorm.ln import Norm
from fnn.fnn_block import FNN
import numpy as np
from Utils.display_weights import show
from pooling.attention_pooling import AttentionPooling
import matplotlib.pyplot as plt

# input_seq=[
#     [
#         [1,0],[1,1]
#     ],
#     [
#         [0,0],[0,1]
#     ],    
#     [
#         [1,1],[0,0]
#     ],
#     [
#         [1,0],[0,1]
#     ],    
#     [
#         [1,1],[1,0]
#     ],
#     [
#         [0,1],[0,0]
#     ],    
#     [
#         [0,0],[1,1]
#     ],
#     [
#         [0,1],[1,0]
#     ],    

# ]
# output_seq=[
#     [
#         [0],[1]
#     ],
#     [
#         [1],[0]
#     ],
#     [
#         [1],[1]
#     ],
#     [
#         [0],[0]
#     ],
#     [
#         [0],[1]
#     ],
#     [
#         [1],[0]
#     ],
#     [
#         [1],[1]
#     ],
#     [
#         [0],[0]
#     ],
# ]

from data import l

amount=20
input_seq=[]
output_seq=[]
for i in range(len(l)-amount):
    input_seq.append(l[i:i+amount])
    output_seq.append([l[i+amount]])

class TransformerBlock:
    def __init__(self, input_seq, output_seq,num_transformer_layers, num_attention_heads,fnn_hidden_size):

        self.layers = []
        self.input_seq=np.array(input_seq)
        self.output_seq=np.array(output_seq)

        # print(self.input_seq,self.input_seq.shape)
        # print(self.output_seq,self.output_seq.shape)

        # self.t_output_weight=np.random.rand(self.input_seq.shape[1],self.output_seq.shape[1])
        self.final_output_matrix1= np.random.rand(self.input_seq.shape[1],self.output_seq.shape[1])
        self.final_output_matrix2= np.random.rand(self.output_seq.shape[0], self.input_seq.shape[0])

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

        self.pooling=AttentionPooling(self.input_seq,final_output_dim=self.output_seq.shape)


    def forward_pass(self,timestep_input,timestep_output):
    # def forward_pass(self):

        # print("input taken -> ",timestep_input)
        # print("output used -> ",timestep_output)

        self.output_from_last_layer=timestep_input
        
        # self.output_from_last_layer=self.input_seq
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

        self.transformation1 = self.output_from_last_layer @ self.final_output_matrix1
        self.transformation2 = self.final_output_matrix2 @ self.transformation1

        # self.loss = self.transformation2 - self.output_seq
        self.loss = self.transformation2 - timestep_output
        self.error = 0.5 * np.sum(self.loss ** 2)

    def backpropagation(self,lr):


        delta_f2 = self.loss @ self.transformation1.T
        gradient = self.loss.T @ self.final_output_matrix2
        delta_f1 = self.output_from_last_layer.T @ gradient.T

        gradient = (self.final_output_matrix1 @ gradient).T

        self.final_output_matrix1 -= delta_f1 * lr
        self.final_output_matrix2 -= delta_f2 * lr

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


nn=TransformerBlock(
    input_seq=input_seq[0],
    output_seq=output_seq[0],
    num_attention_heads=1,
    num_transformer_layers=4,
    fnn_hidden_size=4
)

EPOCHS=500
errors=[]
for _ in range(EPOCHS):

    # nn.forward_pass()
    # # print(f"prediction -> {np.array(nn.transformation2).T} error -> {nn.error}")
    # print("prediction -> ",nn.transformation2,end="\n\n")
    # print("error -> ",nn.error,end="\n\n")
    # nn.backpropagation(lr=0.001)
    # errors.append(nn.error)

    es=0
    print("\n\n==================================\n\n")
    for idx in range(len(input_seq)):
    
        nn.forward_pass(input_seq[idx],output_seq[idx])
        print(f"prediction -> {np.array(nn.transformation2).T} error -> {nn.error}")
        # print("prediction -> ",nn.transformation2,end="\n\n")
        # print("error -> ",nn.error,end="\n\n")
        nn.backpropagation(lr=0.01)
        es+=nn.error

    es /= len(input_seq)
    errors.append(es)

plt.plot(errors)
plt.show()



