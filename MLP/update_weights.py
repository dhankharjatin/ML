def apply_gradients(weights,bias,ln_w,ln_b,gradients,b_gradients,alpha_gradients,beta_gradients,lr=0.01):
    new_weights=[]
    new_bias=[]

    new_alpha_weights=[]
    new_beta_weights=[]


    for i in range(len(weights)):
        new_weights.append(weights[i] - (gradients[i] * lr))

    for i in range(len(bias)):
        new_bias.append(bias[i] - (b_gradients[i][0] * lr))
    
    for i in range(len(ln_w)):
        new_alpha_weights.append(ln_w[i] - (alpha_gradients[i][0] * lr))

    for i in range(len(ln_b)):
        new_beta_weights.append(ln_b[i] - (beta_gradients[i][0] * lr))

    return new_weights,new_bias,new_alpha_weights,new_beta_weights