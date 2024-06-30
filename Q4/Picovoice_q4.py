'''
ASSUMPTIONS:
    -   The blank target label in network outputs is the last value; network_outputs[-1, t]
'''

import numpy as np


def softmax(activations):
    """Softmax activation function."""
    return np.exp(activations) / np.sum(np.exp(activations), axis=0)


def forward_pass(network_outputs, target_labels):
    input_length = network_outputs.shape[1]
    target_length = len(target_labels)
    
    extended_labels = np.zeros(2 * target_length + 1, dtype=int)
    extended_labels[1::2] = target_labels # creates an array with targ labels surrounded by blanks on each side, e.g. ['', a, '', b, '', .....]
    extended_label_length = len(extended_labels)
    
    forward_vars = np.zeros((extended_label_length, input_length))
    
    # Initialization
    forward_vars[0, 0] = network_outputs[-1, 0]  # probability blank label @ time 0
    forward_vars[1, 0] = network_outputs[target_labels[0], 0] # probability non-blank label @ time 0
    for s in range(2, extended_label_length): # set the rest of the values to 0 @ time 0
        forward_vars[s, 0] = 0
    
    # Normalization factor
    C = np.zeros(input_length)
    C[0] = np.sum(forward_vars[:, 0]) # for time 0, sum all the probabilities that we initialized
    forward_vars[:, 0] /= C[0] # normalizes all values for time 0 to prevent numerical underflow
    
    # recursion equations to update forward vars from equation (6) and (7)
    for t in range(1, input_length):
        for s in range(extended_label_length):
            if s == 0:
                forward_vars[s, t] = forward_vars[s, t-1] * network_outputs[-1, t]
            elif s == 1 or extended_labels[s] == extended_labels[s-2]:
                forward_vars[s, t] = (forward_vars[s-1, t-1] + forward_vars[s, t-1]) * network_outputs[extended_labels[s], t]
            else:
                forward_vars[s, t] = (forward_vars[s-2, t-1] + forward_vars[s-1, t-1] + forward_vars[s, t-1]) * network_outputs[extended_labels[s], t]
        
        # normalization of all values for time t to prevent numerical underflow
        C[t] = np.sum(forward_vars[:, t])
        forward_vars[:, t] /= C[t]
    
    return forward_vars, extended_labels, C


def backward_pass(network_outputs, extended_labels, C):
    input_length = network_outputs.shape[1]
    extended_label_length = len(extended_labels)
    
    backward_vars = np.zeros((extended_label_length, input_length))
    
    # Initialization
    backward_vars[-1, -1] = network_outputs[-1, -1]  # probability blank label @ last timestamp
    backward_vars[-2, -1] = network_outputs[extended_labels[-2], -1] # probability non-blank label @ last timestamp
    
    # normalization of all label probs for last timestamp
    backward_vars[:, -1] /= C[-1]
    
    # recursion equations to update forward vars from equation (10) and (11)
    for t in reversed(range(input_length - 1)):
        for s in reversed(range(extended_label_length)):
            if s == extended_label_length - 1:
                backward_vars[s, t] = backward_vars[s, t+1] * network_outputs[-1, t]
            elif s == extended_label_length - 2 or extended_labels[s] == extended_labels[s+2]:
                backward_vars[s, t] = (backward_vars[s, t+1] + backward_vars[s+1, t+1]) * network_outputs[extended_labels[s], t]
            else:
                backward_vars[s, t] = (backward_vars[s, t+1] + backward_vars[s+1, t+1] + backward_vars[s+2, t+1]) * network_outputs[extended_labels[s], t]
        
        # normalization of all label probs for timestamp t
        backward_vars[:, t] /= C[t]
    
    return backward_vars



def ctc_loss(network_outputs, target_labels):
    forward_vars, extended_labels, C = forward_pass(network_outputs, target_labels)
    
    # loss is the negative sum of the logs of the normalization factors
    loss = -np.sum(np.log(C))
    
    return loss


def ctc_grad(network_outputs, target_labels):
    input_length = network_outputs.shape[1]
    num_labels = network_outputs.shape[0]
    
    forward_vars, extended_labels, C = forward_pass(network_outputs, target_labels)
    backward_vars = backward_pass(network_outputs, extended_labels, C)
    
    extended_label_length = len(extended_labels)
    
    grad = np.zeros_like(network_outputs)

    # applied equation (16) to calculate the gradients for backprop
    for t in range(input_length):
        for k in range(num_labels):
            if k == num_labels - 1:  # blank label
                label_indices = [s for s in range(extended_label_length) if extended_labels[s] == 0]
            else:
                label_indices = [s for s in range(extended_label_length) if extended_labels[s] == k]
            
            label_prob = np.sum(forward_vars[s, t] * backward_vars[s, t] for s in label_indices)
            grad[k, t] = network_outputs[k, t] - label_prob / C[t]
    
    return grad