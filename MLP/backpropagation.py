import numpy as np
from Utils.LayerNorm import  create_jacobian
def handel_activation(cal,activation):
    if activation == "ReLU":
        new_cal=[]

        for i in cal:
            if i > 0:
                new_cal.append(1)
            else:
                new_cal.append(0)
        return np.array([new_cal])

    if activation=='sigmoid':
        new_cal=[]
        for i in cal:
            new_cal.append(i*(1-i))

        return np.array([new_cal])


def calculate_gradient(input_seq,f_pass,weights,loss,layer_norm_values,ln_w,z,z_scale,z_norm,activation='ReLU'):
    
    z=z[::-1]
    ln_w=ln_w[::-1]
    f_pass=f_pass[::-1]
    z_norm=z_norm[::-1]
    weights=weights[::-1]
    z_scale=z_scale[::-1]
    layer_norm_values=layer_norm_values[::-1]
    
    graidents=[]
    b_gradients=[]
    beta_gradients=[]
    alpha_gradients=[]

    b_grad=np.array([loss])
    b_gradients.append(b_grad)

    graident=np.array([f_pass[1]]).T @ b_grad
    graidents.append(graident)



    for i in range(len(f_pass)-1):
        if i != len(f_pass)-2:

            #  ==== w.r.t alpha

            b_grad = b_grad @ weights[i].T                                          # multipying with weight 
            b_grad = b_grad * handel_activation(f_pass[i+1],activation=activation)  # handling activation
            beta_gradients.append(b_grad)

            graident = b_grad * z_norm[i]
            alpha_gradients.append(graident)
            
            #  ==== w.r.t w

            b_grad = b_grad * ln_w[i]                                               # multiplying with alpha weights
            b_grad=b_grad @ create_jacobian(layer_norm_values[i])                   # handling layer norm
            b_gradients.append(b_grad)

            graident=b_grad.T @ np.array([f_pass[i+2]])
            graidents.append(graident)
        
        else:
            #  ==== w.r.t alpha

            b_grad = b_grad @ weights[i].T                                          # multipying with weight 
            b_grad = b_grad * handel_activation(f_pass[i+1],activation=activation)  # handling activation
            beta_gradients.append(b_grad)

            graident = b_grad * z_norm[i]
            alpha_gradients.append(graident)
            
            #  ==== w.r.t w

            b_grad = b_grad * ln_w[i]                                               # multiplying with alpha weights
            b_grad=b_grad @ create_jacobian(layer_norm_values[i])                   # handling layer norm
            b_gradients.append(b_grad)

            graident=b_grad.T @ np.array([input_seq])
            graidents.append(graident.T)
        

    return graidents[::-1],b_gradients[::-1],alpha_gradients[::-1],beta_gradients[::-1]

    # #  ==== w.r.t wo

    # b_grad=np.array([loss])
    # graident=np.array([f_pass[1]]).T @ b_grad

    # # print(graident)

    # #  ==== w.r.t alpha 1

    # b_grad = b_grad @ weights[0].T                                          # multipying with weight 
    # b_grad = b_grad * handel_activation(f_pass[1],activation=activation)    # handling activation
    
    # graident = b_grad * z_norm[0]
    
    # # print(graident)

    # #  ==== w.r.t w2

    # b_grad = b_grad * ln_w[0]                                               # multiplying with alpha weights
    # b_grad=b_grad @ create_jacobian(layer_norm_values[0])                   # handling layer norm

    # graident=b_grad.T @ np.array([f_pass[2]])

    # # print(graident)

    # #  ==== w.r.t alpha 2

    # b_grad = b_grad @ weights[1].T                                          # multipying with weight 
    # b_grad = b_grad * handel_activation(f_pass[2],activation=activation)    # handling activation
    
    # graident = b_grad * z_norm[1]

    # # print(graident)

    # #  ==== w.r.t w1

    # b_grad = b_grad * ln_w[1]                                               # multiplying with alpha weights
    # b_grad=b_grad @ create_jacobian(layer_norm_values[1])                   # handling layer norm

    # graident=b_grad.T @ np.array([input_seq])

    # # print(graident)
