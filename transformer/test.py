import matplotlib.pyplot as plt
import numpy as np
import pickle

from post_norm.transformer import TransformerBlock
from weights.assing_weights import assign
from data import l
import pandas as pd
from accelerator import xp

with open("log_return.pkl","rb") as file:
    weights=pickle.load(file)

# for k,v in weights.items():
#     print(k)

# data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
# data=data.loc[:,["open","high","low","close"]]
from dataset.prepare_data import log_returns
data=log_returns


nn = TransformerBlock(
    input_seq=xp.random.rand(2,2),
    output_seq=xp.random.rand(2,2),
    num_attention_heads=weights["num_heads"],
    num_transformer_layers=weights["num_layers"],
    fnn_hidden_size=2
)

assign(nn,weights)


first=1500
last=first+100
output_seq_len=1
num_predicitons=20

# test_seq=data[first:last].values.tolist()
original_seq=data[first:last]
test_seq=data[first:last]
predictions=[]

for i in range(num_predicitons):
    nn.forward_pass(timestep_input=test_seq)
    test_seq=test_seq[1:]
    if output_seq_len == 1:
        test_seq.append(nn.transformation2[0])
        predictions.append(nn.transformation2[0])

    else:
        test_seq.append(nn.transformation2)
        predictions.append(nn.transformation2)


fig,axes=plt.subplots(2,2,figsize=(12,8))

# actual=data[first:last+output_seq_len].values.tolist()
actual=data[first:last+output_seq_len+num_predicitons]
axes[0,0].plot(actual)
# plt.show()

axes[0,1].plot(original_seq)
# axes[0,1].plot(range(len(test_seq),len(test_seq)+len(predictions[0])),predictions[0])
axes[0,1].plot(range(len(test_seq),len(test_seq)+len(predictions)),predictions)
# plt.show()


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
data=data.loc[:,["open","close"]]

og=data[first:last].values.tolist()
data=data[first:last+output_seq_len+num_predicitons].values.tolist()

axes[1,0].plot(data)
# plt.show()


C0,C1=data[last-first]

o=[]
h=[]

for i in predictions:
# for i in predictions[0]:
    o.append(i[0])
    h.append(i[1])



prices1 = C0 * np.exp(np.cumsum(o))
prices2 = C1 * np.exp(np.cumsum(h))

axes[1,1].plot(og)
axes[1,1].plot(range(len(og),len(og)+len(prices1)),prices1)
axes[1,1].plot(range(len(og),len(og)+len(prices2)),prices2)
# plt.show()
plt.tight_layout()
plt.show()