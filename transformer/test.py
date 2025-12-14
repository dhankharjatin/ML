import matplotlib.pyplot as plt
import numpy as np
import pickle

from post_norm.transformer import TransformerBlock
from weights.assing_weights import assign
from data import l
import pandas as pd

with open("weights.pkl","rb") as file:
    weights=pickle.load(file)

# for k,v in weights.items():
#     print(k)

data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
data=data.loc[:,["open","high","low","close"]]

first=830
last=first+100

input_seq=data[first:last].values.tolist()

# input_seq=l[0:30]

nn = TransformerBlock(
    input_seq=input_seq,
    output_seq=input_seq,
    num_attention_heads=weights["num_heads"],
    num_transformer_layers=weights["num_layers"],
    fnn_hidden_size=10
)

assign(nn,weights)


# # test=l[0:30]
test=data[first:last].values.tolist()
predictions=[]

for i in range(1):
    nn.forward_pass(timestep_input=np.array(test))
    # test=test[1:]
    # test.append(nn.transformation2[0])

    # predictions.append(nn.transformation2[0])



actual=data[first:last+30].values.tolist()
plt.plot(actual)
plt.show()

plt.plot(input_seq)
# plt.plot(range(len(test),len(test)+len(predictions)),predictions[0])
plt.plot(range(len(test),len(test)+len(nn.transformation2.tolist())),nn.transformation2.tolist())
plt.show()
