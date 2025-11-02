import numpy as np
def LN(lis,verbose):
    mean=np.average(lis)
    if verbose:
        print("\nmean ",mean)

    variance=0
    deviations=[]

    for i in lis:
        deviation=i-mean    
        deviations.append(deviation)
        variance+=deviation**2

        if verbose:
            print("deviation ",deviation)

    variance=variance/len(lis)
    if verbose:
        print("variance ",variance)

    standard_deviation=variance**(1/2)
    if verbose:
        print("standard_deviation ",standard_deviation,end="\n\n")

    normalized_list=[]
    for i in lis:
        z=(i-mean)/standard_deviation
        normalized_list.append(z)


    return [normalized_list], [deviations,standard_deviation]

def create_jacobian(lis):

    deviations=lis[0]
    sd=lis[1]
    length=len(deviations)

    jacobian=[]

    for i in range(length):
        temp=[]
        for j in range(length):
            if i == j:
                part_1= (1 - (1/length))/sd
            else:
                part_1= (- (1/length))/sd


            part_2= (deviations[i] * deviations[j])/ (length * sd**3)

            temp.append(part_1-part_2)

        jacobian.append(temp)

    return np.array(jacobian)
