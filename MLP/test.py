from forward_pass import calculate_forward_pass

import pickle

# path="ML/MLP/weights/xor.pkl" # XOR
path = "ML/MLP/weights/addition.pkl"  # ADDITION

with open(path, "rb") as f:
    data = pickle.load(f)

input_seq = [10, 12]

# this will remain the same ----------------

weights = data["weights"]
bias = data["bias"]
ln_w = data["ln_w"]
ln_b = data["ln_b"]


result = calculate_forward_pass(input_seq, weights, bias, ln_w, ln_b)

print(result[0][-1])
