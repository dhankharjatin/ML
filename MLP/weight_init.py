import numpy as np

def initialize_weights(input_size,output_size,hidden_size,hidden_layers):
    weights=[]
    bias=[]
    alpha=[]
    beta=[]
    for i in range(hidden_layers+1):

        if i==0:
            w=np.random.rand(input_size,hidden_size)
            b=np.random.rand(hidden_size)
            a=np.ones((1,hidden_size))
            bt=np.zeros((1,hidden_size))
        
            alpha.append(a)
            beta.append(bt)
        
        elif i==hidden_layers:
            w=np.random.rand(hidden_size,output_size)
            b=np.random.rand(output_size)
        else:
            w=np.random.rand(hidden_size,hidden_size)
            b=np.random.rand(hidden_size)
            a=np.ones((1,hidden_size))
            bt=np.zeros((1,hidden_size))
            
            alpha.append(a)
            beta.append(bt)

        weights.append(w)
        bias.append(b)

    return weights,bias,alpha,beta

    # w1=np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]])
    # w2=np.array([[0.1,0.2,0.3],[0.4,0.5,0.6],[0.7,0.8,0.9]])
    # w3=np.array([[.1,.2],[.3,.4],[.5,.6]])

    # b1=np.array([0.1,0.2,0.3])
    # b2=np.array([0.1,0.2,0.3])
    # b3=np.array([0.1,.2])

    # a1=np.array([1,1,1])
    # a2=np.array([1,1,1])

    # be1=np.array([0,0,0])
    # be2=np.array([0,0,0])

    # weights=[w1,w2,w3]
    # bias=[b1,b2,b3]

    # ln_w=[a1,a2]
    # ln_b=[be1,be2]

    # return weights,bias,ln_w,ln_b