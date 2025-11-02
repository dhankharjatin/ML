from weight_init import initialize_weights
from forward_pass import calculate_forward_pass
from backpropagation import calculate_gradient
from update_weights import apply_gradients
import numpy as np

EPOHS=200
# input_seq=[[1,2,3,4],[5,6,7,8]]
# output_seq=[[1,2],[5,6]]

# input_seq=[1,2]
# output_seq=[.1,.2]

# input_seq=[[1,1],[0,0],[0,1],[1,0]]
# output_seq=[[1],[1],[0],[0]]

input_seq=[[1,1],[10,1],[5,6],[1,5],[3,11],[2,3],[5,7],[2,7],[4,15],[4,4],[10,10],[3,3],[4,6],[8,5],[7,1],[4,12],[4,7],[5,9]]
output_seq=[[2],[11],[11],[6],[14],[5],[12],[9],[19],[8],[20],[6],[10],[13],[8],[16],[11],[14]]

print(len(input_seq))
print(len(output_seq))

weights,bias,ln_w,ln_b=initialize_weights(len(input_seq[0]),len(output_seq[0]),hidden_size=10,hidden_layers=1)

# print("weights")
# for i in weights:
#     print(i,end="\n\n")
# print("bias")
# for i in bias:
#     print(i,end="\n\n")
# print("alpha")
# for i in ln_w:
#     print(i,end="\n\n")
# print("beta")
# for i in ln_b:
#     print(i,end="\n\n")

for _ in range(EPOHS):

    for idx in range(len(input_seq)):

        f_pass,mean_deviation_sd,z,z_scale,z_norm=calculate_forward_pass(input_seq[idx],weights,bias,ln_w,ln_b,activation="ReLU")

        loss=f_pass[-1]-output_seq[idx]

        error=[]
        for i in loss:
            error.append(i**2)

        print("PREDICITION ",f_pass[-1],end="   ")
        print("LOSS ",loss,end="   ")
        print("ERROR ",0.5 * sum(error),end="    \n")
        
        gradients,b_gradients,alpha_gradients,beta_gradients=calculate_gradient(input_seq[idx],f_pass,weights,loss,mean_deviation_sd,ln_w,z,z_scale,z_norm)

        # print(f"===== weight gradients ======")
        # for i in gradients:
        #     print(i,end="\n\n")

        # print(f"===== bias gradients ======")
        # for i in b_gradients:
        #     print(i,end="\n\n")

        # print(f"===== alpha gradients ======")
        # for i in alpha_gradients:
        #     print(i,end="\n\n")

        # print(f"===== beta gradients ======")
        # for i in beta_gradients:
        #     print(i,end="\n\n")

        weights,bias,ln_w,ln_b=apply_gradients(weights,bias,ln_w,ln_b,gradients,b_gradients,alpha_gradients,beta_gradients,lr=0.01)
    print("--------------")

data = {}

# # Save weight layers dynamically
# for i, w in enumerate(weights):
#     data[f"w{i+1}"] = w

# for i, b in enumerate(bias):
#     data[f"b{i+1}"] = b

# for i, lw in enumerate(ln_w):
#     data[f"lnw{i+1}"] = lw

# for i, lb in enumerate(ln_b):
#     data[f"lnb{i+1}"] = lb

# # Save everything in one .npz file
# np.savez("MLP/weights/xor.npz", **data)

import pickle

with open("MLP/weights/addition.pkl", "wb") as f:
    pickle.dump({"weights": weights, "bias": bias, "ln_w": ln_w, "ln_b": ln_b}, f)
