# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:28:09 2022

@author: Jan Chodora
@thanks: Martina Kůsová
"""
import random

import math

from scipy.io import arff
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import statistics

def get_data():
    data = arff.loadarff("../data/Diabetic.txt")
    df = pd.DataFrame(data[0])
    return (df["0"],df["1"],df["2"],df["3"],df["4"],df["5"],df["6"], df["7"],df["8"],df["9"],df["10"],df["11"],df["12"],df["13"],df["14"],df["15"],df["16"],df["17"],df["18"]), df["Class"]
#0-7 int, 8-17 double, 18int

def main():
    dim = (19,19, 1)#, 19
    ann = construct_ann(dim)
    print("ANN:",ann,"\n")
    # NN hotovka
    input_data, output_data = get_data()
    for inp in input_data:
        mean = statistics.mean(inp)
        std = statistics.stdev(inp)
        for k in inp.keys():
            inp[k] = ((inp[k] - mean)/std)
    
    
    keys = output_data.keys()
    split_rate_train = 0.75
    split_rate_val = 0.1
    train_keys = []
    validation_keys = []
    test_keys = []
    for k in keys:
        p = random.uniform(0, 1)
        if p < split_rate_train:
            train_keys.append(k)
        else:
            if p < split_rate_train + split_rate_val:
                validation_keys.append(k)
            else:
                test_keys.append(k)
    best_acc, avg_acc, epoch_i, acc = train(ann, dim, train_keys, validation_keys, input_data, output_data)
    plt.plot(acc)
    plt.show()
    print("Training done, ", best_acc, ", avg: ", avg_acc, "epochs: ",epoch_i,", acc", acc)
    #
    test(ann, dim, test_keys, input_data, output_data)
    
def test(ann, dim, test_keys, input_data, output_data):
    accuracies = []
    for k in test_keys:
        targ_out, targ_inp = get_target(dim, output_data, input_data, k)
        out = feed_me(ann, targ_inp, dim)
        accuracy = validation_accuracy(out, targ_out)
        accuracies.append(accuracy)
    avg_acc = float(sum(accuracies))/len(accuracies)
    print("\nTesting done, average accuracy: ",avg_acc)
    return avg_acc


#Marta upravila ukonceni a zaokrouhlouvani
def get_target(dim, output_data, input_data, k):
    targ_out = [0]*dim[len(dim)-1]      # expected output
    targ_out[0] = float(output_data[k])
    
    targ_inp = [0] * dim[0]             # input
    for inp_ix in range(dim[0]):
        #if inp_ix<8 or inp_ix==18:
        #    targ_inp[inp_ix] = int(input_data[inp_ix][k]))
        #else:
        targ_inp[inp_ix] = float(input_data[inp_ix][k])
    return targ_out, targ_inp

def train(ann, dim, train_keys, validation_keys, input_data, output_data):
    EPOCHS = 200
    avg_accuracies=[]
    best_acc = None
    for i in range(EPOCHS):
        #print("\nANN:\n",ann,"\n.\n")
        for k in train_keys:
            targ_out, targ_inp = get_target(dim, output_data, input_data, k)
            out = feed_me(ann, targ_inp, dim)
            
            ann = back_propagation(ann, targ_out, targ_inp)#,dim
        avg_acc = validation(ann, dim, validation_keys, input_data, output_data)
        avg_accuracies.append(avg_acc)
        #    return best_acc, accuracy_avg, i, avg_accuracies
    print("AVG:",avg_accuracies)
    return best_acc,avg_acc,None,avg_accuracies


def validation(ann, dim, validation_keys, input_data, output_data):
    accuracies = []
    for k in validation_keys:
        targ_out, targ_inp = get_target(dim, output_data, input_data, k)
        out = feed_me(ann, targ_inp, dim)
        accuracy = validation_accuracy(out, targ_out)
        accuracies.append(accuracy)
    avg_acc = float(sum(accuracies))/len(accuracies)
    return avg_acc

def validation_accuracy(out, targ_out):
    ERR = 0.48
    correct = 0
    for i in range(len(out)):
        if abs(out[i] - targ_out[i]) < ERR:
            correct += 1.0
    return correct/len(out)

# ----------------------------------------------------------------
#   Functionality:
# ----------------------------------------------------------------

def feed_me(ann, targ_inp, dim):
    layer = targ_inp
    for l in range(len(ann)):
        layer_next = []
        for neuron in ann[l]:
            net = sum_inputs(neuron["weight"], layer)
            output = sigma(net)
            neuron["output"] = output
            #print('\n output:', output)
            
            layer_next.append(neuron["output"])
            
        layer = layer_next
    return layer
    
        
def print_ann(ann):
    for layer in ann:
        for n in layer:
            print(n["weight"])
            

def sum_inputs(weights, inputs):
    x = 0
    for i in range(len(inputs)):
        x += weights[i] * inputs[i]
    return x

def sigma(x):
    sigmoid = 1/(1 + math.e**(-x))
    return sigmoid
# ----------------------------------------------------------------
#   Structure:
# ----------------------------------------------------------------
def construct_ann(dimensions):
    """
    Creates ANN.
    ----------
    @param dimensions:      array of values, one for each layer to be created; 
            first for an input "layer" - will not be created because it does not really exists in our implementation
    """
    ann = [construct_layer(prev_dim=1, actual_dim=dimensions[0])]
    for i in range(1,len(dimensions)):# i as an element of an array
        layer = construct_layer(prev_dim=dimensions[i-1], actual_dim=dimensions[i])
        ann.append(layer)
    return ann


def construct_layer(prev_dim, actual_dim):
    """
    @param prev_dim :       int value; dimension of previous layer
    @param actual_dim :     int value; dimension of previous layer
    --------
    @returns: layer of neurons
    """
    layer = []
    for i in range(actual_dim):
        layer.append(construct_neuron(prev_dim))
    return layer


def construct_neuron(prev_dim):
    """
    @param prev_dim:        int value; dimension of previous layer
    -------
    @returns:               layer of neurons
    """
    neuron_weights = [0.0]*prev_dim
    for i in range(prev_dim):
        neuron_weights[i] = random.uniform(0, 1)
    neuron_output = 0
    return {"weight":neuron_weights,"output":neuron_output}

# NN[
# L  [
#      N {
#            "w":[x y z]
#            "o":x
#       }
#    ]
#   ]
#
#
# ----------------------------------------------------------------
#   Backpropagation:
# ----------------------------------------------------------------

#

#back_prop_2
def back_propagation(ann, target_output, target_input):#, dim len(dim) == len(ann)+1
    deltas_cpy = []
    for lay_ix in range(len(ann)-1, -1, -1):# start, stop, step
        layer = ann[lay_ix]
        if lay_ix >= 1:
            prev_layer = ann[lay_ix-1]
        else:
            prev_layer = target_input
        # for each layer and layer before it
        err_term = 0
        deltas = []
        # for each neuron in the layer
        if lay_ix == len(ann)-1:
            for n_ix in range(len(layer)):
                err_term = error_term_output(t = target_output[n_ix], o = layer[n_ix]["output"])
                deltas.append(err_term)
                propagate_weights(ann, layer, prev_layer, err_term, n_ix, lay_ix)
        else:
            for n_ix in range(len(layer)):
                weights_nxt = []# weights which goes from this neuron to ->next layer
                for next_n in ann[lay_ix+1]:
                    weights_nxt.append(next_n["weight"][n_ix])
                    err_term = error_term_hidden(o_j = layer[n_ix]["output"], deltas_next=deltas_cpy, weights_next=weights_nxt)
                    deltas.append(err_term)
                    propagate_weights(ann, layer, prev_layer, err_term, n_ix, lay_ix)
        #delta = delta_w(o_j = layer[n_ix]["output"], delta_next = err_term)
        
        deltas_cpy = deltas
    return ann

def propagate_weights(ann, source_layer, destination_layer, err_term, n_ix, lay_ix):
    for prev_n_ix in range(len(destination_layer)):# previous layer <-|
        if lay_ix >= 1:
            delt_w = delta_w(destination_layer[prev_n_ix]["output"], err_term)
        else:
            delt_w = delta_w(destination_layer[prev_n_ix], err_term)
        old_w = source_layer[n_ix]["weight"][prev_n_ix]
        new_wn = new_w(old_w, delt_w)
        ann[lay_ix][n_ix]["weight"][prev_n_ix] = new_wn

def trans_next_layer(layer):
    next_layer = [[0.0] * len(layer)] * len(layer[0]["weight"])
    #print(next_layer)
    for i in range(len(layer)):
        for j in range(len(layer[i]["weight"])):
            next_layer[j][i] = layer[i]["weight"][j]
    return next_layer
    

def new_w(old_w, delta_w):
    """
    Calculates value of new weight.
    @param old_w:           old weight for this connction to(?) this neuron
    @param delta_w:         change in the weight
    @return: new weight
    """
    return old_w + delta_w


def delta_w(o_j, delta_next, learning_rate = 0.1):
    """
    Calculates 
    @param o_j:             output of this neuron
    @param delta_next:      error term of the following neuron
    @param learning_rate:   learning constant
    @return: information gain
    """
    return o_j * delta_next * learning_rate


def error_term_hidden(o_j, deltas_next, weights_next):
    """
    delta hidden
    @param o_j:             output of this neuron
    @param deltas_next:     vector; delta is error of following layer's neurons
    @param weights_next:    vector; weight is weight of connection between this neuron and neurons from following layer
    @return: error
    """
    dot_product=0
    for i in range(len(deltas_next)):
        d = deltas_next[i]
        w = weights_next[i]
        dot_product += d * w
    delta = (1 - o_j) * o_j * dot_product
    return delta


def error_term_output(t, o):
    """
    delta output
    @param t:               target output (what I was supposed to get)
    @param o:               output of this neuron (what I got)
    @return: error
    """
    return (t-o)*o*(1-o)



# ----------------------------------------------------------------
#   Executability:
# ----------------------------------------------------------------

if __name__ == "__main__":
    main()


