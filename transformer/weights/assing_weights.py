import numpy as np

def assign(network,weights):
    network.final_output_matrix1=weights["final_output_matrix1"]
    network.final_output_matrix2=weights["final_output_matrix2"]


    for layer_weight,network_layer in zip(weights["layer_weights"],network.layers):

        network_layer[0].q=layer_weight['attention_weights']["q"]
        network_layer[0].k=layer_weight['attention_weights']["k"]
        network_layer[0].v=layer_weight['attention_weights']["v"]
        network_layer[0].wo=layer_weight['attention_weights']["wo"]

        network_layer[1].alpha=layer_weight['ln1_weights']['alpha']
        network_layer[1].beta=layer_weight['ln1_weights']['beta']

        network_layer[2].weights = layer_weight['fnn_weights']['weights']
        network_layer[2].bias = layer_weight['fnn_weights']['bias']

        network_layer[3].alpha=layer_weight['ln2_weights']['alpha']
        network_layer[3].beta=layer_weight['ln2_weights']['beta']
