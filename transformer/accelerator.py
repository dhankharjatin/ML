try : 
    import cupy as xp
    # xp.zeros(1)
    xp.cuda.runtime.getDeviceCount()  # fails if no GPU
    print("-----using cupy")

except Exception as e:
    import numpy as xp
    print("-----using numpy")