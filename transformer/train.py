from post_norm.transformer import TransformerBlock
from weights.save_weights import save
import matplotlib.pyplot as plt
import pandas as pd
from data import l
import numpy as np

from accelerator import xp

# data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
# data=data.loc[:,["open","high","low","close"]]
# data=data[:110].values.tolist()
from dataset.prepare_data import log_returns

data=log_returns

input_size=100
output_size=1

input_seq=[]
output_seq=[]

for i in range(len(data)-input_size-output_size):
    input_seq.append(data[i:i+input_size])
    # output_seq.append([data[i+input_size]])
    output_seq.append(data[i+input_size:i+input_size+output_size])

nn=TransformerBlock(
    input_seq=input_seq[0],
    output_seq=output_seq[0],
    num_attention_heads=2,
    num_transformer_layers=2,   
    fnn_hidden_size=4
)

EPOCHS=1
errors=[]
for _ in range(EPOCHS):

    # nn.forward_pass()
    # # print(f"prediction -> {xp.array(nn.transformation2).T} error -> {nn.error}")
    # print("prediction -> ",nn.transformation2,end="\n\n")
    # print("error -> ",nn.error,end="\n\n")
    # nn.backpropagation(lr=0.001)
    # errors.append(nn.error)

    es=0
    print("\n\n==================================\n\n")
    for idx in range(len(input_seq)):
    
        nn.forward_pass(input_seq[idx])
        nn.backpropagation(lr=0.001,timestep_output=output_seq[idx])
        
        print(f"prediction -> {xp.array(nn.transformation2)} error -> {nn.error}")
        # print("prediction -> ",nn.transformation2,end="\n\n")
        # print("error -> ",nn.error,end="\n\n")
        es+=nn.error

    es /= len(input_seq)
    errors.append(es)

plt.plot(errors)
plt.show()

check=input("save weights y/n? ")
if check == "y":
    print("saved weights")
    save(nn,"log_return")
else:
    print("saved NOT weights")

