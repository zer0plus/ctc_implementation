'''
ASSUMPTIONS:
    -   The blank target label in the network's output is assumed to be the last value; network_outputs[-1, t]  
'''
import os
import torch
import numpy as np
from torch.utils.cpp_extension import load_inline
from cuda_utils import cuda_forward, cuda_backward, cuda_grad, cpp_src


os.environ['CUDA_LAUNCH_BLOCKING']='1'

def load_cuda(cuda_src, cpp_src, funcs, opt=False, verbose=False):
    return load_inline(cuda_sources=[cuda_src], cpp_sources=[cpp_src], functions=funcs,
                    extra_cuda_cflags=["-O2"] if opt else [], verbose=verbose, name="inline_ext")

funcs = ['forward_pass', 'backward_pass', 'cuda_grad']
CUDA_passes = load_cuda(cuda_forward+cuda_backward+cuda_grad, cpp_src, funcs)



def softmax(activations):
    return np.exp(activations) / np.sum(np.exp(activations), axis=0)


# utilizes equations (6)&(7) from the paper for forward pass calculations
def forward_pass(network_outputs, target_labels):
    input_length = network_outputs.shape[1]
    target_length = len(target_labels)
    
    extended_labels = np.zeros(2 * target_length + 1, dtype=int)
    extended_labels[1::2] = target_labels # creates an array with targ labels surrounded by blanks on each side, e.g. ['', a, '', b, '', .....]
    extended_label_length = len(extended_labels)
    
    forward_vars = torch.zeros((extended_label_length, input_length), dtype=torch.float32, device='cuda')
    C = torch.zeros(input_length, dtype=torch.float32, device='cuda')

    network_outputs_cuda = network_outputs.cuda()
    extended_labels_cuda = torch.from_numpy(extended_labels).cuda()

    CUDA_passes.forward_pass(network_outputs_cuda.data_ptr(), extended_labels_cuda.data_ptr(), 
                    forward_vars.data_ptr(), C.data_ptr(),
                    input_length, extended_label_length, network_outputs.shape[0])

    return forward_vars.cpu().numpy(), extended_labels, C.cpu().numpy()


# utilizes equations (10)&(11) from the paper for backward pass calculations
def backward_pass(network_outputs, extended_labels, C):
    input_length = network_outputs.shape[1]
    extended_label_length = len(extended_labels)
    
    backward_vars = torch.zeros((extended_label_length, input_length), dtype=torch.float32, device='cuda')
    network_outputs_cuda = network_outputs.cuda()
    extended_labels_cuda = torch.from_numpy(extended_labels).cuda()

    CUDA_passes.backward_pass(network_outputs_cuda.data_ptr(), extended_labels_cuda.data_ptr(), 
                    backward_vars.data_ptr(), C.data_ptr(),
                    input_length, extended_label_length, network_outputs.shape[0])
        
    return backward_vars.cpu().numpy()


def ctc_loss(network_outputs, target_labels):
    forward_vars, extended_labels, C = forward_pass(network_outputs, target_labels)
    
    # loss is the negative sum of the logs of the normalization factors
    loss = -np.sum(np.log(C))
    
    return loss


# utilizes equation (16) from the paper for gradient calculation
def ctc_grad(network_outputs, target_labels):
    input_length = network_outputs.shape[1]
    num_labels = network_outputs.shape[0]
    
    forward_vars, extended_labels, C = forward_pass(network_outputs, target_labels)
    backward_vars = backward_pass(network_outputs, extended_labels, C)
    
    extended_label_length = len(extended_labels)
    
    network_outputs_cuda = torch.from_numpy(network_outputs).cuda()
    extended_labels_cuda = torch.from_numpy(extended_labels).cuda()
    forward_vars_cuda = torch.from_numpy(forward_vars).cuda()
    backward_vars_cuda = torch.from_numpy(backward_vars).cuda()
    C_cuda = torch.from_numpy(C).cuda()
    
    grad_cuda = torch.zeros_like(network_outputs_cuda)

    CUDA_passes.cuda_grad(
        network_outputs_cuda.data_ptr(),
        extended_labels_cuda.data_ptr(),
        forward_vars_cuda.data_ptr(),
        backward_vars_cuda.data_ptr(),
        C_cuda.data_ptr(),
        grad_cuda.data_ptr(),
        input_length,
        extended_label_length,
        num_labels
    )
    
    return grad_cuda.cpu().numpy()