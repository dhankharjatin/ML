from forward_pass import calculate_forward_pass

import pickle

path="MLP/weights/xor.pkl" # XOR
# path="MLP/weights/addition.pkl" # ADDITION

with open(path,"rb") as f:
    data = pickle.load(f)

input_seq=[1,0]

# this will remain the same ----------------

weights=data['weights']
bias=data['bias']
ln_w=data['ln_w']
ln_b=data['ln_b']


result=calculate_forward_pass(input_seq,weights,bias,ln_w,ln_b)

print(result[0][-1])