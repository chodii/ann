# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:28:09 2022

@author: Jan Chodora
"""
import random

import math

from scipy.io import arff
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import statistics

import io

def dictionarize_X_y(X, y, y_range = range(1,6,1)):
    len_X = len(X)# since array should be fixed, O(1); just felt wrong not to save
    Xy = {yi:[[] for _ in range(len_X)] for yi in y_range}# prepare dictionary
    for i in range(len(y)):# fill dictionary
        yi = int(y[i])
        for j in range(len_X):
            Xy[yi][j].append(X[j][i])
    return Xy
    
def draw_data(Xy, colours=None):
    for yi in Xy.keys():# draw each class separately
        plt.scatter(Xy[yi][0], Xy[yi][1], label=str(yi), color=colours[yi] if colours is not None else colours)
    plt.legend(loc="upper left")
    plt.show()
    
def draw_out_data(results, colours, plt3d=None):
    if plt3d is not None:
        for result in results:# draw each class separately
            plt3d.scatter(result[0][0], result[0][1], result[0][2], color=colours[result[2]], edgecolors=colours[result[1]])
        #return
    for result in results:# draw each class separately
        plt.scatter(result[0][0], result[0][1], color=colours[result[2]], edgecolors=colours[result[1]])
    
def get_alter_data():
    raw_data = arff.loadarff("../data/Diabetic.txt")
    df = pd.DataFrame(raw_data[0])
    #0-7 int, 8-17 double, 18int
    data = (df["0"],df["1"],df["2"],df["3"],df["4"],df["5"],df["6"], df["7"],df["8"],df["9"],df["10"],df["11"],df["12"],df["13"],df["14"],df["15"],df["16"],df["17"],df["18"]), df["Class"]
    return data

def get_data(filepath="../data/tren_data2___08.txt", columns=3):
    lines = None
    with open(filepath) as f:
        lines = f.readlines()
    data = [[] for _ in range(columns)]
    for line in lines:
        line_postex = np.loadtxt(io.StringIO(line))
        for i in range(len(data)):
            data[i].append(line_postex[i])
    return (data[0], data[1]), data[2]


def main():
    
    
    dim = (2,5,10, 5)#, 19
    EPOCHS = 300
    learning_rate = 0.8
    ann = construct_ann(dim)
    print("ANN:",ann,"\n")
    # NN hotovka
    input_data, output_data = get_data()
    draw_data(dictionarize_X_y(input_data, output_data))
    for inp in input_data:
        mean = statistics.mean(inp)# EX
        std = statistics.stdev(inp)# varX
        #for k in inp.keys():
        #minp = min(inp)
        #manp = max(inp)
        for k in range(len(inp)):
            inp[k] = ((inp[k] - mean)/std)# norm
            #inp[k] = (inp[k] - minp)/(-minp + manp)
    print(input_data)
    
    keys = np.arange(len(output_data)).tolist()#output_data.keys()
    train_keys, validation_keys, test_keys = split_train_val_test(keys)
    
    best_acc, avg_acc, acc = train(ann, dim, train_keys, validation_keys, input_data, output_data, EPOCHS=EPOCHS, learning_rate=learning_rate)
    plt.plot(acc)
    plt.show()
    print("Training done, ", best_acc, ", avg: ", avg_acc, ", acc", np.mean(acc), max(acc))
    #
    test_subjects = test(ann, dim, test_keys, input_data, output_data)
    print("ANN dimension:", dim)
    
    #3D
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    
    colours=["brown", "green", "purple", "blue", "red", "black", "white"]
    take_cover = 1000
    pts = []
    for _ in range(take_cover):
        pt = [random.uniform(-2.0,2.0), random.uniform(-2, 2)] 
        out = feed_me(ann, pt, dim)
        
        vectout = vectorize_neuron(out)
        pt3d = [pt[0], pt[1], max(vectout)]
        scout = scalaratize_output(vectout)
        pts.append((pt3d, scout, -2 if scout == -1 else -1))
    draw_out_data(pts, colours, ax)
    
    
    results = [ 0 for _ in range(len(test_subjects))]
    j = 0
    for k, prediction, target in test_subjects:
        position = [input_data[i][k] for i in range(dim[0])]
        results[j] = (position, prediction, target)
        j += 1
    draw_out_data(results, colours[:dim[-1]])
    
    
    plt.show()
    

    # /!\  /!\  /!\ Str
    # /!\  /!\  /!\ Strange behaviour detected /!\  /!\  /!\, inspect:
    # print("incorrect predictions:\n".join(str(incorrect_pred_keys[i][1])+" != "+str(incorrect_pred_keys[i][2]) + " on [".join(str(incorrect_predictions[d][i]) for d in range(dim[0]))+"], \n" for i in range(len(incorrect_pred_keys))))
    

def split_train_val_test(keys):
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
    return train_keys, validation_keys, test_keys

def test(ann, dim, test_keys, input_data, output_data):
    accuracies = []
    results = []
    for k in test_keys:
        targ_out, targ_inp = get_target(dim, output_data, input_data, k)
        out = feed_me(ann, targ_inp, dim)
        
        vectout = vectorize_neuron(out)
        accuracy = prediction_accuracy(vectout, targ_out)
        accuracies.append(accuracy)
        
        results.append((k, scalaratize_output(vectout), scalaratize_output(targ_out)))
    avg_acc = float(sum(accuracies))/len(accuracies)
    print("\nTesting done, average accuracy: ",avg_acc)
    print(results)
    return results

def scalaratize_output(out):
    epsilon = sum(out)/2
    for i in range(len(out)):
        if out[i] > epsilon:
            return i
    return -1


def get_target(dim, output_data, input_data, k):
    """ 
    Vectorization of input/output
    --------
    @return: vector, vector
    """
    targ_out = one_hot_encoding(output_data[k], dim[-1])
    targ_inp = [0] * dim[0]
    for inp_ix in range(dim[0]):
        targ_inp[inp_ix] = float(input_data[inp_ix][k])
    return targ_out, targ_inp

def train(ann, dim, train_keys, validation_keys, input_data, output_data, EPOCHS = 100, learning_rate=0.3, batch_size = 10):
    
    avg_accuracies=[]
    best_acc = None
    b = 0
    target_outputs = []
    for i in range(EPOCHS):
        print(".", end="")
        #outpus = []
        for k in train_keys:
            targ_out, targ_inp = get_target(dim, output_data, input_data, k)
            out = feed_me(ann, targ_inp, dim)
            #outpus.append(out)
            target_outputs.append(targ_out)
            
            b += 1
            if b%batch_size == 0:
                #ann = propagation(ann, target_outputs, learning_rate)
                target_outputs = []
                
            ann = back_propagation(ann, targ_out, learning_rate)#,dim
        avg_acc = validation(ann, dim, validation_keys, input_data, output_data)
        avg_accuracies.append(avg_acc)
        #    return best_acc, accuracy_avg, i, avg_accuracies
    print("AVG:",np.mean(avg_accuracies), max(avg_accuracies))
    return best_acc,avg_acc,avg_accuracies


def validation(ann, dim, validation_keys, input_data, output_data):
    accuracies = []
    for k in validation_keys:
        targ_out, targ_inp = get_target(dim, output_data, input_data, k)
        out = feed_me(ann, targ_inp, dim)
        
        vectout = vectorize_neuron(out)
        accuracy = prediction_accuracy(vectout, targ_out)
        accuracies.append(accuracy)
    avg_acc = float(sum(accuracies))/len(accuracies)
    return avg_acc

def prediction_accuracy(outvect, targ_out):
    """
    Accuracy of a single prediction made.
    --------
    @param out: output layer
    @param targ_out: vector
    --------
    returns: scalar
    """
    outnorm = 1/sum(outvect)
    
    err_vect = [0] * len(targ_out)
    for i in range(len(targ_out)):
        err_vect[i] = abs(outvect[i]*outnorm - targ_out[i])
    accuracy = (2 - sum(err_vect))/2
    return accuracy

def vectorize_neuron(out):
    outvect = [0] * len(out)
    for i in range(len(out)):
        outvect[i] = out[i]["output"]
    return outvect

# ----------------------------------------------------------------
#   Functionality:
# ----------------------------------------------------------------

def feed_me(ann, targ_inp, dim):
    for i in range(len(targ_inp)):
        ann[0][i]["output"] = targ_inp[i]
    
    for l in range(1, len(ann), 1):
        for neuron in ann[l]:
            net = sum_inputs(neuron["weight"], ann[l-1])
            neuron["output"] = sigma(net)
    return ann[len(ann)-1]
    
        
def print_ann(ann):
    for layer in ann:
        for n in layer:
            print(n["weight"])
            

def sum_inputs(weights, inputs):
    x = 0
    for i in range(len(inputs)):
        x += weights[i] * inputs[i]["output"]
    x += weights[-1]# +bias
    return x

def sigma(x):
    sigmoid = 1/(1 + math.e**(-x))
    return sigmoid
# ----------------------------------------------------------------
#   Structure:
# ----------------------------------------------------------------

def one_hot_encoding(num, vect_len):
    ix = int(num) - 1
    v = [0] * vect_len
    v[ix] = 1
    return v

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
    neuron_weights = [0.0]* (prev_dim +1)# +1 for bias
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

#back_prop_2
def back_propagation(ann, target_output, learning_rate):#, dim len(dim) == len(ann)+1
    """
        network propagation
    """
    deltas_cpy = []
    for lay_ix in range(len(ann)-1, 1, -1):# start, stop, step
        layer = ann[lay_ix]
        prev_layer = ann[lay_ix-1]
        # for each layer and layer before it
        err_term = 0
        deltas = []
        # for each neuron in the layer
        for n_ix in range(len(layer)):
            if lay_ix == len(ann)-1:
                err_term = error_term_output(t = target_output[n_ix], o = layer[n_ix]["output"])
            else:
                err_term = error_term_hidden(o_j = layer[n_ix]["output"], deltas_next=deltas_cpy, layer_next=ann[lay_ix+1], neuron_ix=n_ix)
            deltas.append(err_term)
            propagate_weights(ann, layer, prev_layer, err_term, n_ix, lay_ix, learning_rate)
        #delta = delta_w(o_j = layer[n_ix]["output"], delta_next = err_term)
        deltas_cpy = deltas
    return ann

def propagate_weights(ann, source_layer, destination_layer, err_term, n_ix, lay_ix, learning_rate):
    """
        neuron's weights propagation
    """
    for prev_n_ix in range(len(destination_layer)):# previous layer <-|
        if lay_ix >= 1:
            delt_w = delta_w(destination_layer[prev_n_ix]["output"], err_term, learning_rate)
        else:
            delt_w = delta_w(destination_layer[prev_n_ix]["output"], err_term, learning_rate)
        old_w = source_layer[n_ix]["weight"][prev_n_ix]
        ann[lay_ix][n_ix]["weight"][prev_n_ix] = old_w + delt_w
    
    delt_wb = delta_w(1, err_term, learning_rate)
    old_wb = source_layer[n_ix]["weight"][-1]
    ann[lay_ix][n_ix]["weight"][-1] = old_wb + delt_wb
    

def delta_w(o_j, delta_next, learning_rate = 0.5):
    """
    Calculates 
    @param o_j:             output of this neuron
    @param delta_next:      error term of the following neuron
    @param learning_rate:   learning constant
    @return: information gain
    """
    return o_j * delta_next * learning_rate


def error_term_hidden(o_j, deltas_next, layer_next, neuron_ix):
    """
    delta hidden
    @param o_j:             output of this neuron
    @param deltas_next:     vector; delta is error of following layer's neurons
    @param weights_next:    vector; weight is weight of connection between this neuron and neurons from following layer
    @return: error
    """
    if len(deltas_next) != len(layer_next):
        print("Error: ", deltas_next, neuron_ix, len(layer_next))
    dot_product=0
    for i in range(len(deltas_next)):
        d = deltas_next[i]
        w = layer_next[i]["weight"][neuron_ix]
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
#   Execute:
# ----------------------------------------------------------------

if __name__ == "__main__":
    main()


