import numpy as np

class AttentionPooling:
    def __init__(self,inp,final_output_dim):
        self.q = np.random.rand(final_output_dim[0],inp.shape[1])
        self.final_output_weight=np.random.rand(inp.shape[1],final_output_dim[1])


    def softmax(matrix):
        softmax_matrix=[]
        for row in matrix:
            row=row-np.max(row)
            row=np.exp(row)
            row=row/np.sum(row)

            softmax_matrix.append(row)

        return np.array(softmax_matrix)

    def softmax_jacobian(softmax_row):
        softmax_row = softmax_row.reshape(-1, 1)  # column vector
        J = np.diagflat(softmax_row) - np.dot(softmax_row, softmax_row.T)
        return J

    def forward(self,output_from_last_layer):
        self.output_from_last_layer = output_from_last_layer

        self.score=self.q @ output_from_last_layer.T
        self.scaled_score = self.score / output_from_last_layer.shape[1] ** (1/2)
        self.softmax_scaled_score= AttentionPooling.softmax(self.scaled_score)
        self.pooled=self.softmax_scaled_score @ output_from_last_layer
        
        self.output = self.pooled @ self.final_output_weight

    def backpropagation(self,loss):
        self.delta_output_w= self.pooled.T @ loss 

        delta_q = loss @ self.final_output_weight.T
        delta_q = delta_q @ self.output_from_last_layer.T

        new_dq=[]
        result = np.array([AttentionPooling.softmax_jacobian(row) for row in self.softmax_scaled_score])
        for idx,i in enumerate(result):
            new_dq.append(delta_q[idx] @ i)

        delta_q=np.array(new_dq)
        delta_q /= self.output_from_last_layer.shape[1] ** (1/2)
        
        self.gradient_to_next_layer=delta_q.T @ self.q
        self.delta_q = delta_q @ self.output_from_last_layer


    def update_weights(self,lr):

        self.final_output_weight -= self.delta_output_w * lr
        self.q -= self.delta_q * lr