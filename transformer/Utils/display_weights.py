def show(layers):
    for layer in layers:
        ln_1=layer[0]
        attention_block=layer[1]
        ln_2=layer[2]
        fnn=layer[3]

        print("-------- layer norm ----------")
        print("alpha")
        for w in ln_1.alpha:
            print(w,end="\n\n")

        print("beta")
        for w in ln_1.beta:
            print(w,end="\n\n")

        print("-------- attention block ----------")
        print(f"q => {attention_block.q}\n\nk => {attention_block.k}\n\nv => {attention_block.v}\n\nwo => {attention_block.wo}")
        
        print("-------- layer norm 2 ----------")
        print("alpha")
        for w in ln_2.alpha:
            print(w,end="\n\n")


        print("beta")
        for w in ln_2.beta:
            print(w,end="\n\n")


        print("-------- fnn ----------")
        print("weights")
        for w in fnn.weights:
            print(w,end="\n\n")

        print("bias")
        for w in fnn.bias:
            print(w,end="\n\n")
