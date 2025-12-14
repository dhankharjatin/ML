from post_norm.transformer import TransformerBlock
from weights.save_weights import save
import matplotlib.pyplot as plt
import pandas as pd
from data import l
import numpy as np

data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
data=data.loc[:,["open","high","low","close"]]

data=data[:1000].values.tolist()

amount=100
input_seq=[]
output_seq=[]
for i in range(len(data)-amount-30):
    input_seq.append(data[i:i+amount])
    # output_seq.append([data[i+amount]])
    output_seq.append(data[i+amount:i+amount+30])

nn=TransformerBlock(
    input_seq=input_seq[0],
    output_seq=output_seq[0],
    num_attention_heads=2,
    num_transformer_layers=10,   
    fnn_hidden_size=4
)

EPOCHS=3
errors=[]
for _ in range(EPOCHS):

    # nn.forward_pass()
    # # print(f"prediction -> {np.array(nn.transformation2).T} error -> {nn.error}")
    # print("prediction -> ",nn.transformation2,end="\n\n")
    # print("error -> ",nn.error,end="\n\n")
    # nn.backpropagation(lr=0.001)
    # errors.append(nn.error)

    es=0
    print("\n\n==================================\n\n")
    for idx in range(len(input_seq)):
    
        nn.forward_pass(input_seq[idx])
        nn.backpropagation(lr=0.00001,timestep_output=output_seq[idx])
        
        print(f"prediction -> {np.array(nn.transformation2)} error -> {nn.error}")
        # print("prediction -> ",nn.transformation2,end="\n\n")
        # print("error -> ",nn.error,end="\n\n")
        es+=nn.error

    es /= len(input_seq)
    errors.append(es)

plt.plot(errors)
plt.show()

save(nn)

# og=l[0:amount]
# test=l[0:amount]

# for i in range(50):
#     nn.forward_pass(timestep_input=np.array(test))
#     test=test[1:]
#     test.append(tuple(nn.transformation2[0]))

# plt.plot(og)
# plt.plot(range(len(og),len(og)+len(test)),test)
# plt.show()

# og=l[20:20+amount]
# test=l[20:20+amount]

# for i in range(50):
#     nn.forward_pass(timestep_input=np.array(test))
#     test=test[1:]
#     test.append(tuple(nn.transformation2[0]))

# plt.plot(og)
# plt.plot(range(len(og),len(og)+len(test)),test)
# plt.show()
