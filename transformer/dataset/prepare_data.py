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
    # per.append(difference)

    close_return = math.log(next[3]) - math.log(current[3])
    open_return = math.log(next[0]) - math.log(current[0])
    log_returns.append([close_return,open_return])

    # print(difference, log_return)

# plt.plot(log_returns)
# plt.show()



# C0=1.5026
# prices = C0 * np.exp(np.cumsum(l))

# data=pd.read_csv("transformer/dataset/SOL_LARGE.csv")
# only_close=data.loc[:,["close"]]
# only_close=only_close[:200]


# plt.plot(only_close)
# plt.plot(prices)
# plt.show()


