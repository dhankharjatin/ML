import matplotlib.pyplot as plt
import numpy as np
import pickle

from post_norm.transformer import TransformerBlock
from weights.assing_weights import assign
from data import l
import pandas as pd
from accelerator import xp

with open("weights.pkl","rb") as file:
    weights=pickle.load(file)

# for k,v in weights.items():
#     print(k)

data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
data=data.loc[:,["open","high","low","close"]]



nn = TransformerBlock(
    input_seq=xp.random.rand(2,2),
    output_seq=xp.random.rand(2,2),
    num_attention_heads=weights["num_heads"],
    num_transformer_layers=weights["num_layers"],
    fnn_hidden_size=10
)

assign(nn,weights)


first=830
last=first+100
output_seq_len=1
num_predicitons=1

test_seq=data[first:last].values.tolist()
predictions=[]

for i in range(num_predicitons):
    nn.forward_pass(timestep_input=xp.array(test_seq))
    test_seq=test_seq[1:]
    if output_seq_len == 1:
        test_seq.append(nn.transformation2[0])
        predictions.append(nn.transformation2[0])

    else:
        test_seq.append(nn.transformation2)
        predictions.append(nn.transformation2)

actual=data[first:last+output_seq_len].values.tolist()
plt.plot(actual)
plt.show()

plt.plot(test_seq)
# plt.plot(range(len(test),len(test)+len(predictions)),predictions[0])
plt.plot(range(len(test_seq),len(test_seq)+len(predictions)),predictions)
plt.show()
