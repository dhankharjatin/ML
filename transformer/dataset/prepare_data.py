import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
data=data.loc[:,["open","high","low","close"]]

data=data.values.tolist()

per=[]
log_returns=[]
for i in range(len(data)-1):

    current=data[i]
    next=data[i+1]

    # difference = ((next[3] - current[3]) / current[3]) *100

    log_return = math.log(next[3]) - math.log(current[3])

    # per.append(difference)
    log_returns.append(log_return)
    # print(difference, log_return)

# plt.plot(per)
# plt.plot(l)
# plt.show()



# C0=1.5026
# prices = C0 * np.exp(np.cumsum(l))

# data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
# only_close=data.loc[:,["close"]]
# only_close=only_close[:200]


# plt.plot(only_close)
# plt.plot(prices)
# plt.show()


