import cupy as c
import numpy as np
import time

limit=200

# numpy ==============================

def numpy_test(shape,limit):
    start=time.time()
    gradient=np.random.rand(shape[0],shape[1])
    for i in range(limit):
        random_mat=np.random.rand(shape[0],shape[1])
        gradient @= random_mat
    
    print(f"\n\nnumpy done in {time.time()-start} \n\n")

# cupy ==============================

def cupy_test(shape,limit):
    start=time.time()
    gradient=c.random.rand(shape[0],shape[1])
    for i in range(limit):
        random_mat=c.random.rand(shape[0],shape[1])
        gradient @= random_mat
    
    print(f"\n\ncupy done in {time.time()-start} \n\n")
    input("adfasdf")

shapes=[(5,5),(100,100),(264,264),(512,512),(1024,1024),(4096,4096)]
# limit=[]

# for i in shapes:
#     print("for ",i)
#     numpy_test(i,200)
#     cupy_test(i,200)

cupy_test((4096*4,4096*4),200)