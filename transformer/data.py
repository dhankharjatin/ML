import math
import matplotlib.pyplot as plt

l=[]
v=20
for i in range(61):
    s=math.sin(i) *2
    c=math.cos(i) *2
    # v/=1.01

    mul=(s*v) /4

    l.append((s,c,mul))


if __name__ == "__main__":

    plt.plot(l)
    plt.show()
