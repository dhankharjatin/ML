# from attention.attention_block import AttentionBlock

input_seq = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
# # input_seq = [[1, 1.1, 1.2, 1.3], [1.1, 1.2,1.3, 1.4], [1.2, 1.3, 1.4, 1.5]]

# nn = AttentionBlock(input_seq=input_seq, num_heads=2, verbose=True)
# nn.weight_init()
# nn.forward_pass()
# # nn.backpropagation()

from LayerNorm.ln import Norm
import numpy as np

f = Norm()

norm_lis, values = f.forward(np.array([input_seq[0]]))

print(norm_lis, end="\n\n")
print(values, end="\n\n")

for i in norm_lis:
    print(i)
for i in values:
    print(i)

j_matrix = f.create_jacobian(values)

print()
for i in j_matrix:
    print(i, end="\n\n")
