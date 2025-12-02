import numpy as np

class AttentionBlock:
    def __init__(self,num_heads,input_seq,verbose=False):
        
        self.input_seq=np.array(input_seq)
        self.num_heads=num_heads
        self.verbose=verbose
    
    def softmax(self,matrix):
        softmax_matrix=[]
        for row in matrix:
            row=row-np.max(row)
            row=np.exp(row)
            row=row/np.sum(row)

            softmax_matrix.append(row)

        return softmax_matrix

    def weight_init(self):

        rows ,column=self.input_seq.shape
        self.q = np.random.rand(column, column)
        self.k = np.random.rand(column, column)
        self.v = np.random.rand(column, column)
        self.wo = np.random.rand(column, column)

        # self.q = np.ones((column, column))
        # self.k = np.ones((column, column))
        # self.v = np.ones((column, column))
        # self.wo = np.ones((column, column))

        # self.q = np.array(
        #     [
        #         [0.1, 0.2, 0.3, 0.4],
        #         [0.1, 0.2, 0.3, 0.4],
        #         [0.1, 0.2, 0.3, 0.4],
        #         [0.1, 0.2, 0.3, 0.4],
        #     ],dtype=np.float64
        # )
        # self.k = np.array(
        #     [
        #         [0.1, 0.2, 0.3, 0.4],
        #         [0.5, 0.6, 0.7, 0.8],
        #         [0.9, 1.0, 1.1, 1.2],
        #         [1.3, 1.4, 1.5, 1.6],
        #     ],dtype=np.float64
        # )
        # self.v = np.array(
        #     [
        #         [0.1, 0.1, 0.1, 0.1],
        #         [0.2, 0.2, 0.2, 0.2],
        #         [0.3, 0.3, 0.3, 0.3],
        #         [0.4, 0.4, 0.4, 0.4],
        #     ],dtype=np.float64
        # )

        # self.wo = np.array(
        #     [
        #         [1, 0, 0, 0],
        #         [0, 1, 0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1],
        #     ],dtype=np.float64
        # )

        if self.verbose:
            print(f"============ initial weights ============\n\n q => {self.q}\n\nk => {self.k}\n\nv => {self.v}\n\nwo => {self.wo}")


    def forward_pass(self,input_from_last_layer):
        
        self.input_from_last_layer=np.array(input_from_last_layer)

        # Linear projection
        self.Q = self.input_from_last_layer @ self.q
        self.K = self.input_from_last_layer @ self.k
        self.V = self.input_from_last_layer @ self.v

        if self.verbose:
            print(f"============ Linear projection ============\n\n inp.q => {self.Q}\n\ninp.k => {self.K}\n\ninp.v => {self.V}\n\n")
        
        rows, columns = self.Q.shape

        if columns % self.num_heads != 0:
            print("invalid number of heads, Defaulting to num_head = 1")
            self.num_heads = 1

        self.Qs = np.array_split(self.Q, self.num_heads, axis=1)
        self.Ks = np.array_split(self.K, self.num_heads, axis=1)
        self.Vs = np.array_split(self.V, self.num_heads, axis=1)

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
        self.masking_matrix=np.triu(np.ones((rows,rows)),k=1)*-1e9
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

        self.combined_matrix = np.concat(self.output_matrix, axis=1)
        self.output_MHA = self.combined_matrix @ self.wo

        if self.verbose:
            print(f"============ ConCat ============\n\n combined matrix => {self.combined_matrix}\n\noutputMHA (combined.wo) => {self.output_MHA}")

    
    def softmax_jacobian(softmax_row):
        softmax_row = softmax_row.reshape(-1, 1)  # column vector
        J = np.diagflat(softmax_row) - np.dot(softmax_row, softmax_row.T)
        return J


    def backpropagation(self,gradient_from_last_layer):

        print("\n\n============= bakpropagation ======================= \n\n")

        self.delta=gradient_from_last_layer.T @ self.combined_matrix
        print("gradient for wo -> ",self.delta,end="\n\n")
        self.delta=self.combined_matrix.T @ gradient_from_last_layer 
        print("gradient for wo -> ",self.delta,end="\n\n")

        gradient=gradient_from_last_layer @ self.wo
        
        self.gradient_v=[]
        self.gradient_k=[]
        self.gradient_q=[]

        self.gqs=[]
        self.gks=[]
        self.gvs=[]

        split=np.array_split(gradient,self.num_heads,axis=1)
        for idx,i in enumerate(split):

            g_v=(i.T @ np.array(self.all_softmax_masked_score[idx])).T

            gradient =i @ self.Vs[idx].T

            
            # backdroping though softmax =====================================================

            softmax_delta=[]
            jacobians = np.array([AttentionBlock.softmax_jacobian(row) for row in self.all_softmax_masked_score[idx]])
            for idx_1,j in enumerate(jacobians):
                softmax_delta.append(gradient[idx_1].T @ j)

            softmax_delta=np.array(softmax_delta)

            # ===================================================================================

            gradient @= softmax_delta 
            gradient /=self.D_k 

            g_q= gradient @ self.Ks[idx]
            self.gradient_q.append(self.input_from_last_layer.T @ g_q)

            g_k = gradient @ self.Qs[idx]
            self.gradient_k.append(self.input_from_last_layer.T @ g_k)

            self.gradient_v.append(self.input_from_last_layer.T @ g_v)

            self.gqs.append(g_q)
            self.gks.append(g_k)
            self.gvs.append(g_v)


        self.total_gradient_q=np.concat(self.gradient_q,axis=1)
        self.total_gradient_v=np.concat(self.gradient_v,axis=1)
        self.total_gradient_k=np.concat(self.gradient_k,axis=1)

        print("gradient for q",self.total_gradient_q,end="\n\n")
        print("gradient for k ",self.total_gradient_k,end="\n\n")
        print("gradient for v",self.total_gradient_v,end="\n\n")

        self.gqs=np.concat(self.gqs,axis=1)
        self.gks=np.concat(self.gks,axis=1)
        self.gvs=np.concat(self.gvs,axis=1)

        self.gradient_to_next_layer = (self.gqs @ self.q ) + (self.gks @ self.q ) + (self.gvs @ self.q ) 

        print("gradient for next layer -> ",self.gradient_to_next_layer)


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
