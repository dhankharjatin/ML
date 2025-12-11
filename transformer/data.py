import math
import matplotlib.pyplot as plt

l=[]
v=20
for i in range(201):
    s=math.sin(i)
    c=math.cos(i)
    v/=1.1

    mul=s*v*c

    l.append((s,c+2,v,mul))


if __name__ == "__main__":

    plt.plot(l)
    plt.show()
