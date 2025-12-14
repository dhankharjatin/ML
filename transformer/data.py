import math
import matplotlib.pyplot as plt
import random

limit=121

l=[]

up=True
tri=0
for i in range(limit):
    s=math.sin(i) *2
    c=math.cos(i) *2

    r=5
    if i%5==0:
        r=10
        if i %10 ==0:
            r=0

    if i% 30 ==0:
        up = not up

    if up:
        tri-=0.5

    else:
        tri+=0.5


    l.append((s,c,r,tri))


if __name__ == "__main__":

    plt.plot(l)
    plt.show()
