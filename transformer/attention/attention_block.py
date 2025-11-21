import numpy as np

class AttentionBlock:
    def __init__(self,input_seq,num_heads,verbose=False):
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

        rows, columns = self.input_seq.shape
        
        self.q = np.random.rand(columns, columns)
        self.k = np.random.rand(columns, columns)
        self.v = np.random.rand(columns, columns)
        self.wo = np.random.rand(columns, columns)

        self.q = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
                [0.1, 0.2, 0.3, 0.4],
            ]
        )
        self.k = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5, 1.6],
            ]
        )
        self.v = np.array(
            [
                [0.1, 0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2, 0.2],
                [0.3, 0.3, 0.3, 0.3],
                [0.4, 0.4, 0.4, 0.4],
            ]
        )

        self.wo = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )


        # Linear projection
        self.Q = self.input_seq @ self.q
        self.K = self.input_seq @ self.k
        self.V = self.input_seq @ self.v

        if self.verbose:
            print(f"============ initial weights ============\n\n q => {self.q}\n\nk => {self.k}\n\nv => {self.v}\n\n")


    def forward_pass(self):

        # Linear projection
        self.Q = self.input_seq @ self.q
        self.K = self.input_seq @ self.k
        self.V = self.input_seq @ self.v

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

        self.output_matrix = []
        
        for idx in range(len(self.Qs)):

            self.score = self.Qs[idx] @ self.Ks[idx].T
            self.scaled_score = self.score / self.D_k ** (1 / 2)
            self.softmax_scaled_score = self.softmax(self.scaled_score)
            self.output = self.softmax_scaled_score @ self.Vs[idx]
            self.output_matrix.append(self.output)
        
            if self.verbose:
                print(f"============ Forward pass for head {idx} ============\n\n score (q.k) => {self.score}\n\nscaled_score => {self.scaled_score}\n\nsoftmax => {self.softmax_scaled_score}\n\n attention_score (softmax.v) => {self.output}")

        self.combined_matrix = np.concat(self.output_matrix, axis=1)
        self.output_MHA = self.combined_matrix @ self.wo

        if self.verbose:
            print(f"============ ConCat ============\n\n combined matrix => {self.combined_matrix}\n\noutputMHA (combined.wo) => {self.output_MHA}")

    def backpropagation(self):

        layer_norm = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ]
    )
        self.gradient_wo=self.output_MHA.T @ layer_norm
        print(self.gradient_wo)

        f=layer_norm @ self.wo
        f=np.array_split(f,2,axis=1)
        print(f[0] @ self.Vs[0].T)
        print()