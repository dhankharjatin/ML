from attention.attention_block import AttentionBlock
input_seq = [[1, 2, 3, 4], [5, 6,7, 8], [9, 10, 11, 12]]
# input_seq = [[1, 1.1, 1.2, 1.3], [1.1, 1.2,1.3, 1.4], [1.2, 1.3, 1.4, 1.5]]

nn=AttentionBlock(input_seq=input_seq,num_heads=2,verbose=True)
nn.weight_init()
nn.forward_pass()
# nn.backpropagation()