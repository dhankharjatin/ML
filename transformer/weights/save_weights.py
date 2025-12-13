def save(network):

    print(network.final_output_matrix1)
    print(network.final_output_matrix2)

    for layer in network.layers:
        attention,ln1,fnn,ln2 = layer

        print("---attention")
        print(attention.q)
        print(attention.k)
        print(attention.v)
        print(attention.wo)

        print("---ln1")
        print(ln1.alpha)
        print(ln1.beta)

        print("---fnn")
        print("w",fnn.weights)    
        print("b",fnn.bias)    

        print("---ln2")
        print(ln2.alpha)
        print(ln2.beta)