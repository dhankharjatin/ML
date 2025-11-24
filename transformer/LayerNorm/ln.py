import numpy as np


class Norm:
    def forward(self, matrix, verbose=False):

        self.normalized_matrix = []
        self.d_and_sd = []

        for lis in matrix:
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

    def create_jacobian(self, matrix):

        self.jacobian_matrix = []
        for lis in matrix:
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
