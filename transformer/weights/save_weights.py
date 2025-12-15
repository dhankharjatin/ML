import json
import pickle
def save(network,file_name):

    weights={}

    weights["num_layers"]=network.num_transformer_layers
    weights["num_heads"]=network.num_attention_heads

    weights['final_output_matrix1'] = network.final_output_matrix1
    weights['final_output_matrix2'] = network.final_output_matrix2

    layer_weights=[]
    for idx,layer in enumerate(network.layers):
        attention,ln1,fnn,ln2 = layer

        attention_weights={
            "q":attention.q,
            "k":attention.k,
            "v":attention.v,
            "wo":attention.wo,
            }

        ln1_weights={
            "alpha" : ln1.alpha,
            "beta" : ln1.beta,
        }

        fnn_weights = {
            "weights":fnn.weights,
            "bias":fnn.bias

        }

        ln2_weights={
            "alpha" : ln2.alpha,
            "beta" : ln2.beta,
        }

        layer_weights.append(
            {
            "attention_weights":attention_weights,
            "ln1_weights": ln1_weights,
            "fnn_weights": fnn_weights,
            "ln2_weights": ln2_weights,
        }
        )

        # weights[f"layer{idx}"]={
        #     "attention_weights":attention_weights,
        #     "ln1_weights": ln1_weights,
        #     "fnn_weights": fnn_weights,
        #     "ln2_weights": ln2_weights,
        # }

    weights["layer_weights"] = layer_weights

    with open(f"{file_name}.pkl","wb") as file:
        # json.dump(weights,file)
        pickle.dump(weights,file)