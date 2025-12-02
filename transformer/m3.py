from attention.attention_block import AttentionBlock
from LayerNorm.ln import Norm
from fnn.fnn_block import FNN
import numpy as np
from Utils.display_weights import show
# input_seq = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# input_seq = [[1, 1], [1, 1]]
# input_seq = [[1.11, 1.45, 0.99], [0.99, 1.32, 1.44], [0.88, 0.94, 1.22]]

input_seq = [[1.11, 1.45, 0.99,1.00], [0.99, 1.32, 1.44,1.1], [0.88, 0.94, 1.22,.77]]
output_seq=[[1,2],[2,3],[3,4]]

# input_seq = [[0,0], [1,1], [1,0],[0,1]]
# output_seq=[[1],[1],[0],[0]]

class TransformerBlock:
    def __init__(self, input_seq, output_seq,num_transformer_layers, num_attention_heads,fnn_hidden_size,fnn_hidden_layers):

        self.layers = []
        self.input_seq=np.array(input_seq)
        self.output_seq=np.array(output_seq)

        self.t_output_weight=np.random.rand(self.input_seq.shape[1],self.output_seq.shape[1])

        for _ in range(num_transformer_layers):

            ln_1 = Norm(self.input_seq)
            ln_1.weight_init()

            attention_block = AttentionBlock(
                num_heads=num_attention_heads,
                input_seq=self.input_seq,
            )
            attention_block.weight_init()

            ln_2 = Norm(self.input_seq)
            ln_2.weight_init()

            fnn=FNN(self.input_seq,hidden_size=fnn_hidden_size,hidden_layers=fnn_hidden_layers)
            fnn.weight_init()

            self.layers.append([ln_1,attention_block,ln_2,fnn])

        # show(self.layers)

    def forward_pass(self):

        self.output_from_last_layer=self.input_seq
        self.forward_pass_values=[]


        for layer in self.layers:
            ln_1 : Norm=layer[0]
            attention : AttentionBlock=layer[1]
            ln_2 : Norm=layer[2]
            fnn : FNN=layer[3]


            ln_1.forward(self.output_from_last_layer)
            ln_1.scale()

            attention.forward_pass(input_from_last_layer=ln_1.scaled_matrix)
            
            residual_connection_1=self.output_from_last_layer+attention.output_MHA

            ln_2.forward(residual_connection_1)
            ln_2.scale()

            fnn.forward(input_from_last_layer=ln_2.scaled_matrix)

            residual_connection_2=residual_connection_1+fnn.f_pass_values[-1][0]

            self.output_from_last_layer=residual_connection_2

            self.forward_pass_values.append(
                [
                    ln_1.normalized_matrix,
                    ln_1.scaled_matrix,
                    attention.output_MHA,
                    residual_connection_1,
                    ln_2.normalized_matrix,
                    ln_2.scaled_matrix,
                    fnn.f_pass_values[-1],
                    residual_connection_2,
                ]
            )
        self.t_output = self.output_from_last_layer @ self.t_output_weight
        self.loss = self.t_output - self.output_seq

        self.error = 0.5 * np.sum(self.loss ** 2)

    def backpropagation(self, lr):

        # ----- FINAL LINEAR LAYER -----
        delta_t_out_weight = self.output_from_last_layer.T @ self.loss
        self.t_output_weight -= lr * delta_t_out_weight

        # gradient flowing into last transformer block
        gradient = self.loss @ self.t_output_weight.T


        # ----- BACKWARD THROUGH ALL TRANSFORMER LAYERS -----
        for layer_idx, layer in enumerate(self.layers[::-1]):

            fnn: FNN = layer[-1]
            ln_2: Norm = layer[-2]
            attention_block: AttentionBlock = layer[-3]
            ln_1: Norm = layer[-4]

            # ===================================================
            #   RESIDUAL CONNECTION 2
            #   residual_2 = residual_1 + fnn_output
            # ===================================================

            # gradient flows BOTH to fnn_output AND residual_1
            grad_to_fnn_output = gradient        # branch 1
            grad_to_residual_1 = gradient.copy() # branch 2 (identity path)

            # ----- FNN -----
            fnn.backpropagation(gradient_from_last_layer=grad_to_fnn_output)
            fnn.update_weights(lr=lr)

            gradient = fnn.gradient_to_next_layer + grad_to_residual_1
            # now gradient = dL/d(residual_1)


            # ===================================================
            #   LAYER NORM 2
            # ===================================================
            ln_2.create_jacobian()
            ln_2.backpropagation(
                gradient_from_last_layer=gradient,
                jacobian_matrix=ln_2.jacobian_matrix
            )
            ln_2.update_weights(lr=lr)

            gradient = ln_2.gradient_to_next_layer
            # now gradient = dL/d(residual_1 BEFORE LN2)


            # ===================================================
            #   RESIDUAL CONNECTION 1
            #   residual_1 = input_prev + mha_output
            # ===================================================

            grad_to_mha_output = gradient         # branch 1
            grad_to_input_prev = gradient.copy()  # branch 2 (identity path)


            # ----- ATTENTION BLOCK -----
            attention_block.backpropagation(
                gradient_from_last_layer=grad_to_mha_output
            )
            attention_block.update_weights(lr=lr)

            gradient = attention_block.gradient_to_next_layer + grad_to_input_prev
            # now gradient = dL/d(input_before_ln1)


            # ===================================================
            #   LAYER NORM 1
            # ===================================================
            ln_1.create_jacobian()
            ln_1.backpropagation(
                gradient_from_last_layer=gradient,
                jacobian_matrix=ln_1.jacobian_matrix
            )
            ln_1.update_weights(lr=lr)

            gradient = ln_1.gradient_to_next_layer
            # final gradient passed into next lower layer



t = TransformerBlock(input_seq=input_seq,output_seq=output_seq,num_attention_heads=2,num_transformer_layers=2,fnn_hidden_size=3,fnn_hidden_layers=2)

EPOCHS=50
for _ in range(EPOCHS):
    t.forward_pass()
    print(f"PREDICTION -> {np.array(t.t_output).T} LOSS -> {np.array(t.loss).T} ERROR -> {t.error}")
    t.backpropagation(lr=0.01)

# for i in t.forward_pass_values:

#     print("\n\n------------ ------------------\n")
#     for j in i:
#         print(j,end="\n\n")

# print("output => ",t.t_output)
# print(t.loss)
# print(t.error)

# ==========================================================================================

# f=FNN(input_seq=np.array(input_seq),hidden_size=1,hidden_layers=2,verbose=True)
# f.weight_init()
# f.forward(input_from_last_layer=np.array(input_seq))
# f.backpropagation(gradient_from_last_layer=np.eye(4,2))
# f.update_weights(lr=0.01)


# ==========================================================================================

# l=Norm(input_seq=np.array(input_seq),verbose=True)
# l.weight_init()
# l.forward(input_from_last_layer=np.array(input_seq))
# l.scale()
# l.create_jacobian()
# l.backpropagation(np.eye(3,4),l.jacobian_matrix)
# l.update_weights(lr=0.1)

# print(l.jacobian_matrix)
# print(l.delta)
# print(l.gradient_to_next_layer)

# ==========================================================================================

# a=AttentionBlock(num_heads=2,input_seq=np.array(input_seq),verbose=True)
# a.weight_init()
# a.forward_pass(input_from_last_layer=np.array(input_seq))
# a.backpropagation(gradient_from_last_layer=np.eye(3,4)) 
# a.update_weights(lr=0.1)