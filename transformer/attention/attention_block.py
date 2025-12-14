import numpy as np
from accelerator import xp

class AttentionBlock:
    def __init__(self,num_heads,input_seq,verbose=False):
        
        self.input_seq=xp.array(input_seq)
        self.num_heads=num_heads
        self.verbose=verbose
    
    def softmax(self,matrix):
        softmax_matrix=[]
        for row in matrix:
            row=row-xp.max(row)
            row=xp.exp(row)
            row=row/xp.sum(row)

            softmax_matrix.append(row)

        return xp.array(softmax_matrix)

    def weight_init(self):

        rows ,column=self.input_seq.shape
        self.q = xp.random.rand(column, column)
        self.k = xp.random.rand(column, column)
        self.v = xp.random.rand(column, column)
        self.wo = xp.random.rand(column, column)

        # self.q = xp.ones((column, column))
        # self.k = xp.ones((column, column))
        # self.v = xp.ones((column, column))
        # self.wo = xp.ones((column, column))

        # self.q = xp.array(
        #     [
        #         [0.1, 0.2, 0.3, 0.4],
        #         [0.1, 0.2, 0.3, 0.4],
        #         [0.1, 0.2, 0.3, 0.4],
        #         [0.1, 0.2, 0.3, 0.4],
        #     ],dtype=xp.float64
        # )
        # self.k = xp.array(
        #     [
        #         [0.1, 0.2, 0.3, 0.4],
        #         [0.5, 0.6, 0.7, 0.8],
        #         [0.9, 1.0, 1.1, 1.2],
        #         [1.3, 1.4, 1.5, 1.6],
        #     ],dtype=xp.float64
        # )
        # self.v = xp.array(
        #     [
        #         [0.1, 0.1, 0.1, 0.1],
        #         [0.2, 0.2, 0.2, 0.2],
        #         [0.3, 0.3, 0.3, 0.3],
        #         [0.4, 0.4, 0.4, 0.4],
        #     ],dtype=xp.float64
        # )

        # self.wo = xp.array(
        #     [
        #         [1, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1],
        #     ],dtype=xp.float64
        # )
        # self.q = xp.array([[0.54996149, 0.39209668, 0.02231746, 0.48168687],
        #       [0.17846079, 0.59506052, 0.27767094, 0.99438220],
        #       [0.78372351, 0.93071373, 0.60868734, 0.25713706],
        #       [0.86374547, 0.20451310, 0.52022287, 0.73113249]])

        # self.k = xp.array([[0.13179803, 0.86064070, 0.64492262, 0.47987069],
        #             [0.16408382, 0.83949395, 0.37309877, 0.95698553],
        #             [0.22570978, 0.58066413, 0.69773053, 0.58552252],
        #             [0.48725987, 0.32354720, 0.01374808, 0.40714439]])

        # self.v = xp.array([[0.09519453, 0.29168806, 0.47696261, 0.27997762],
        #             [0.69538272, 0.22538543, 0.30028171, 0.37976110],
        #             [0.04170441, 0.17910670, 0.22816673, 0.01213948],
        #             [0.70658640, 0.27623120, 0.85991468, 0.13645307]])

        # self.wo = xp.array([[0.48194419, 0.75491935, 0.32472981, 0.17631607],
        #             [0.64027037, 0.78707261, 0.55516965, 0.01710959],
        #             [0.26693266, 0.88208908, 0.16231246, 0.99274457],
        #             [0.48128538, 0.19762023, 0.36148378, 0.64238365]])

        if self.verbose:
            print(f"============ initial weights ============\n\n q => {self.q}\n\nk => {self.k}\n\nv => {self.v}\n\nwo => {self.wo}")


    def forward_pass(self,input_from_last_layer):
        
        self.input_from_last_layer=xp.array(input_from_last_layer)

        # Linear projection
        self.Q = self.input_from_last_layer @ self.q
        self.K = self.input_from_last_layer @ self.k
        self.V = self.input_from_last_layer @ self.v

        if self.verbose:
            print(f"============ Linear projection ============\n\n ixp.q => {self.Q}\n\nixp.k => {self.K}\n\nixp.v => {self.V}\n\n")
        
        rows, columns = self.Q.shape

        if columns % self.num_heads != 0:
            print("invalid number of heads, Defaulting to num_head = 1")
            self.num_heads = 1

        self.Qs = xp.array_split(self.Q, self.num_heads, axis=1)
        self.Ks = xp.array_split(self.K, self.num_heads, axis=1)
        self.Vs = xp.array_split(self.V, self.num_heads, axis=1)

        self.D_k = columns/self.num_heads

        if self.verbose:
            print(f"\n\n======================= splits ====================\n\n")
            print("Qs")
            for i in self.Qs:
                print(i,end="\n\n")
            print("Ks")
            for i in self.Ks:
                print(i,end="\n\n")
            print("Vs")
            for i in self.Vs:
                print(i,end="\n\n")

        self.output_matrix = []
        
        # creating mask
        self.masking_matrix=xp.triu(xp.ones((rows,rows)),k=1)*-1e9
        if self.verbose:
            print("masking matrix -> ",self.masking_matrix,end="\n\n")

        self.all_softmax_masked_score=[]

        for idx in range(len(self.Qs)):

            self.score = self.Qs[idx] @ self.Ks[idx].T
            self.scaled_score = self.score / self.D_k ** (1 / 2)
            self.masked_score = self.scaled_score + self.masking_matrix
            self.softmax_masked_score = self.softmax(self.masked_score)
            self.attention_score = self.softmax_masked_score @ self.Vs[idx]
            self.output_matrix.append(self.attention_score)

            self.all_softmax_masked_score.append(self.softmax_masked_score)
            
            if self.verbose:
                print(f"============ Forward pass for head {idx} ============\n\n score (q.k) => {self.score}\n\nscaled_score => {self.scaled_score}\n\nmasked_score => {self.masked_score}\n\nsoftmax => {self.softmax_masked_score}\n\n attention_score (softmax.v) => {self.attention_score}")

        self.combined_matrix = xp.concatenate(self.output_matrix, axis=1)
        self.output_MHA = self.combined_matrix @ self.wo

        if self.verbose:
            print(f"============ ConCat ============\n\n combined matrix => {self.combined_matrix}\n\noutputMHA (combined.wo) => {self.output_MHA}")

    
    def softmax_jacobian(softmax_row):
        softmax_row = softmax_row.reshape(-1, 1)  # column vector
        J = xp.diagflat(softmax_row) - xp.dot(softmax_row, softmax_row.T)
        return J


    def backpropagation(self,gradient_from_last_layer):

        # print("\n\n============= bakpropagation ======================= \n\n")

        # self.delta=gradient_from_last_layer.T @ self.combined_matrix
        # print("gradient for wo -> ",self.delta,end="\n\n")
        self.delta=self.combined_matrix.T @ gradient_from_last_layer 
        # print("gradient for wo -> ",self.delta,end="\n\n")

        gradient=gradient_from_last_layer @ self.wo.T
        
        self.gradient_v=[]
        self.gradient_k=[]
        self.gradient_q=[]

        self.gqs=[]
        self.gks=[]
        self.gvs=[]

        split=xp.array_split(gradient,self.num_heads,axis=1)
        for idx,i in enumerate(split):

            g_v=(i.T @ xp.array(self.all_softmax_masked_score[idx])).T

            gradient =i @ self.Vs[idx].T
            
            # backdroping though softmax =====================================================

            softmax_delta=[]
            jacobians = xp.array([AttentionBlock.softmax_jacobian(row) for row in self.all_softmax_masked_score[idx]])
            for idx_1,j in enumerate(jacobians):
                softmax_delta.append(gradient[idx_1].T @ j)

            softmax_delta=xp.array(softmax_delta)

            # ===================================================================================

            softmax_delta /=self.D_k ** (1/2) 

            g_q= softmax_delta @ self.Ks[idx]
            self.gradient_q.append(self.input_from_last_layer.T @ g_q)

            g_k = softmax_delta.T @ self.Qs[idx]
            self.gradient_k.append(self.input_from_last_layer.T @ g_k)

            self.gradient_v.append(self.input_from_last_layer.T @ g_v)

            self.gqs.append(g_q)
            self.gks.append(g_k)
            self.gvs.append(g_v)


        self.total_gradient_q=xp.concatenate(self.gradient_q,axis=1)
        self.total_gradient_v=xp.concatenate(self.gradient_v,axis=1)
        self.total_gradient_k=xp.concatenate(self.gradient_k,axis=1)

        # print("gradient for q",self.total_gradient_q,end="\n\n")
        # print("gradient for k ",self.total_gradient_k,end="\n\n")
        # print("gradient for v",self.total_gradient_v,end="\n\n")

        self.gqs=xp.concatenate(self.gqs,axis=1)
        self.gks=xp.concatenate(self.gks,axis=1)
        self.gvs=xp.concatenate(self.gvs,axis=1)

        self.gradient_to_next_layer = (self.gqs @ self.q.T ) + (self.gks @ self.k.T ) + (self.gvs @ self.v.T ) 

        # print("gradient for next layer -> ",self.gradient_to_next_layer)


    def update_weights(self,lr):
        self.q -= self.total_gradient_q * lr
        self.k -= self.total_gradient_k * lr
        self.v -= self.total_gradient_v * lr

        self.wo -= self.delta * lr

        
        # print("\n\n========== updated weight ===============\n")
        # print("q => ",self.q,end="\n\n")
        # print("k => ",self.k,end="\n\n")
        # print("v => ",self.v,end="\n\n")
        # print("wo => ",self.wo,end="\n\n")
