import numpy as np


class Norm:
    def __init__(self,input_seq,verbose=False):
        self.input_seq= input_seq
        self.verbose=verbose

    def weight_init(self):
        rows, columns = self.input_seq.shape

        self.alpha = np.ones((rows, columns))
        self.beta = np.zeros((rows, columns))

    def forward(self,input_from_last_layer):

        self.input_from_last_layer = input_from_last_layer
        self.normalized_matrix = []
        self.d_and_sd = []

        for lis in self.input_from_last_layer:
            mean = np.average(lis)
            if self.verbose:
                print("\nmean ", mean)

            variance = 0
            deviations = []

            for i in lis:
                deviation = i - mean
                deviations.append(deviation)
                variance += deviation**2

                if self.verbose:
                    print("deviation ", deviation)

            variance = variance / len(lis)
            if self.verbose:
                print("variance ", variance)

            standard_deviation =( variance + 1e-5 ) ** (1 / 2)
            if self.verbose:
                print("standard_deviation ", standard_deviation, end="\n\n")

            normalized_list = []
            for i in lis:
                z = (i - mean) / (standard_deviation ) 
                normalized_list.append(z)

            self.normalized_matrix.append(normalized_list)
            self.d_and_sd.append([deviations, standard_deviation])

    def scale(self):
        self.scaled_matrix = (np.array(self.normalized_matrix) * self.alpha) + self.beta
        
        if self.verbose:
            print("---------------- norm ---------------")
            print("normalized matrix => ",self.normalized_matrix,end="\n\n")

            print("---------------- using weights ---------------")
            print("alpha => ",self.alpha,end="\n\n")
            print("beta => ",self.beta,end="\n\n")
            print("---------------- scaling ---------------")
            print("scaled matrix => ",self.scaled_matrix,end="\n\n")

    def create_jacobian(self):

        self.jacobian_matrix = []
        for lis in self.d_and_sd:
            deviations = lis[0]
            sd = lis[1]
            length = len(deviations)

            jacobian = []

            for i in range(length):
                temp = []
                for j in range(length):
                    if i == j:
                        part_1 = (1 - (1 / length)) / (sd)
                    else:
                        part_1 = (-(1 / length)) / (sd)

                    part_2 = (deviations[i] * deviations[j]) / ((length * sd**3))
                    temp.append(part_1 - part_2)

                jacobian.append(temp)

            self.jacobian_matrix.append(jacobian)
        self.jacobian_matrix=np.array(self.jacobian_matrix)

    def backpropagation(self,gradient_from_last_layer,jacobian_matrix):

        self.gradient_from_last_layer=gradient_from_last_layer
        
        self.gradient=self.gradient_from_last_layer * self.normalized_matrix


        self.delta=self.gradient_from_last_layer * self.alpha
        self.gradient_to_next_layer=[]
        for idx,i in enumerate(jacobian_matrix):
            self.gradient_to_next_layer.append(i @ self.delta[idx].T)

        self.gradient_to_next_layer=np.array(self.gradient_to_next_layer)

        if self.verbose:
            print("---------------- bp ---------------")
            print("gradient alpha -> ", self.gradient)
            print("gradient beta -> ", self.gradient_from_last_layer,end="\n\n")
    
    def update_weights(self,lr):
        self.alpha -= self.gradient * lr
        self.beta -= self.gradient_from_last_layer * lr

        # print("\n\n========== updated weight ===============\n")
        # print("alpha => ",self.alpha,end="\n\n")
        # print("beta => ",self.beta,end="\n\n")
