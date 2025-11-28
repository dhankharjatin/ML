import numpy as np


class Norm:
    def __init__(self,input_seq):
        self.input_seq= input_seq

    def weight_init(self):
        rows, columns = self.input_seq.shape

        self.alpha = np.ones((rows, columns))
        self.beta = np.zeros((rows, columns))

    def forward(self,input_from_last_layer, verbose=False):

        self.input_from_last_layer = input_from_last_layer
        self.normalized_matrix = []
        self.d_and_sd = []

        for lis in self.input_from_last_layer:
            mean = np.average(lis)
            if verbose:
                print("\nmean ", mean)

            variance = 0
            deviations = []

            for i in lis:
                deviation = i - mean
                deviations.append(deviation)
                variance += deviation**2

                if verbose:
                    print("deviation ", deviation)

            variance = variance / len(lis)
            if verbose:
                print("variance ", variance)

            standard_deviation = variance ** (1 / 2)
            if verbose:
                print("standard_deviation ", standard_deviation, end="\n\n")

            normalized_list = []
            for i in lis:
                z = (i - mean) / standard_deviation
                normalized_list.append(z)

            self.normalized_matrix.append(normalized_list)
            self.d_and_sd.append([deviations, standard_deviation])

    def scale(self):
        self.scaled_matrix = (np.array(self.normalized_matrix) * self.alpha) + self.beta

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
                        part_1 = (1 - (1 / length)) / sd
                    else:
                        part_1 = (-(1 / length)) / sd

                    part_2 = (deviations[i] * deviations[j]) / (length * sd**3)
                    temp.append(part_1 - part_2)

                jacobian.append(temp)

            self.jacobian_matrix.append(jacobian)
        self.jacobian_matrix=np.array(self.jacobian_matrix)

    def backpropagation(self,gradient_from_last_layer,jacobian_matrix):
        gradient=gradient_from_last_layer * self.input_from_last_layer
        print(gradient,end="\n\n")
        print(jacobian_matrix,end="\n\n")

        gradient_to_next_layer=[]
        for idx,i in enumerate(jacobian_matrix):
            gradient_to_next_layer.append(i @ gradient[idx].T)

        gradient_to_next_layer=np.array(gradient_to_next_layer)
        print(gradient_to_next_layer)