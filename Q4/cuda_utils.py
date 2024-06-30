cuda_init = r'''
#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

inline unsigned int cdiv(unsigned int a, unsigned int b) { return (a + b - 1) / b;}
'''


cuda_forward = cuda_init + r'''
__global__ void forward_pass_kernel(float* network_outputs, int* extended_labels, float* forward_vars, float* C, 
                                    int input_length, int extended_label_length, int num_labels) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (s < extended_label_length && t < input_length) {
        int idx = s*input_length + t;

        if (t == 0) {
            if (s == 0) 
                forward_vars[idx] = network_outputs[(num_labels-1)*input_length];
            else if (s == 1)
                forward_vars[idx] = network_outputs[extended_labels[1]*input_length];
            else
                forward_vars[idx] = 0.0f;

            // normalization for t=0
            __syncthreads();
            if (s == 0) {
                float sum = 0.0f;
                for (int i = 0; i < extended_label_length; i++) {
                    sum += forward_vars[i*input_length];
                }
                C[0] = sum;
            }
            __syncthreads();
            forward_vars[idx] /= C[0];
        }
        else {
            // Recursive equations for t > 0
            if (s == 0)
                forward_vars[idx] = forward_vars[idx-input_length] * network_outputs[(num_labels-1)*input_length+t];
            else if (s == 1 || extended_labels[s] == extended_labels[s-2])
                forward_vars[idx] = (forward_vars[idx-input_length-1] + forward_vars[idx-input_length]) * network_outputs[extended_labels[s]*input_length+t];
            else
                forward_vars[idx] = (forward_vars[idx-2*input_length-1] + forward_vars[idx-input_length-1] + forward_vars[idx-input_length]) * network_outputs[extended_labels[s]*input_length+t];

            // normalization for t > 0  
            __syncthreads();
            if (s == 0) {
                float sum = 0.0f;
                for (int i = 0; i < extended_label_length; i++) {
                    sum += forward_vars[i*input_length + t];
                }
                C[t] = sum;
            }
            __syncthreads();
            forward_vars[idx] /= C[t];
        }
    }
}

void forward_pass(float* network_outputs_ptr, int* extended_labels_ptr, float* forward_vars_ptr, float* C_ptr,
                int input_length, int extended_label_length, int num_labels) {
    dim3 threads(16, 16);
    dim3 blocks(cdiv(extended_label_length, threads.x), cdiv(input_length, threads.y));
    
    forward_pass_kernel<<<blocks, threads>>>(network_outputs_ptr, extended_labels_ptr, forward_vars_ptr, C_ptr, 
                                            input_length, extended_label_length, num_labels);
'''



cuda_backward = cuda_init + r'''
__global__ void backward_pass_kernel(float* network_outputs, int* extended_labels, float* backward_vars, float* C, 
                                    int input_length, int extended_label_length, int num_labels) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    int t = input_length - 1 - (blockIdx.y * blockDim.y + threadIdx.y); // reverse mapping to ensure a backwards loop
    
    if (s < extended_label_length && t >= 0) {
        int idx = s*input_length + t;

        if (t == input_length - 1) {
            if (s == extended_label_length - 1)
                backward_vars[idx] = network_outputs[(num_labels-1)*input_length + t];
            else if (s == extended_label_length - 2)
                backward_vars[idx] = network_outputs[extended_labels[s]*input_length + t];
            else
                backward_vars[idx] = 0.0f;

            // normalization for t=input_length-1
            __syncthreads();
            backward_vars[idx] /= C[t];
        }
        else {
            // Recursive equations for t < input_length-1
            if (s == extended_label_length - 1)
                backward_vars[idx] = backward_vars[idx+input_length] * network_outputs[(num_labels-1)*input_length+t];
            else if (s == extended_label_length - 2 || extended_labels[s] == extended_labels[s+2])
                backward_vars[idx] = (backward_vars[idx+input_length] + backward_vars[idx+input_length+1]) * network_outputs[extended_labels[s]*input_length+t];
            else
                backward_vars[idx] = (backward_vars[idx+input_length] + backward_vars[idx+input_length+1] + backward_vars[idx+input_length+2]) * network_outputs[extended_labels[s]*input_length+t];

            // normalization for t < input_length-1
            __syncthreads();  
            backward_vars[idx] /= C[t];
        }
    }
}

void backward_pass(float* network_outputs_ptr, int* extended_labels_ptr, float* backward_vars_ptr, float* C_ptr,
                int input_length, int extended_label_length, int num_labels) {
    dim3 threads(16, 16);
    dim3 blocks(cdiv(extended_label_length, threads.x), cdiv(input_length, threads.y));
    
    backward_pass_kernel<<<blocks, threads>>>(network_outputs_ptr, extended_labels_ptr, backward_vars_ptr, C_ptr, 
                                            input_length, extended_label_length, num_labels);
}
'''

cuda_grad = cuda_init + r'''
__global__ void cuda_grad_kernel(float* network_outputs, int* extended_labels, 
                                float* forward_vars, float* backward_vars, float* C,
                                float* grad, int input_length, int extended_label_length, int num_labels) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (t < input_length && k < num_labels) {
        float label_prob = 0.0f;
        
        for (int s = 0; s < extended_label_length; s++) {
            if ((k == num_labels - 1 && extended_labels[s] == 0) || 
                (k < num_labels - 1 && extended_labels[s] == k)) {
                int idx = s * input_length + t;
                label_prob += forward_vars[idx] * backward_vars[idx];
            }
        }
        
        grad[k * input_length + t] = network_outputs[k * input_length + t] - label_prob / C[t];
    }
}

void cuda_grad(float* network_outputs_ptr, int* extended_labels_ptr, 
            float* forward_vars_ptr, float* backward_vars_ptr, float* C_ptr,
            float* grad_ptr, int input_length, int extended_label_length, int num_labels) {
    dim3 threads(16, 16);
    dim3 blocks(cdiv(input_length, threads.x), cdiv(num_labels, threads.y));
    
    cuda_grad_kernel<<<blocks, threads>>>(network_outputs_ptr, extended_labels_ptr,
                                        forward_vars_ptr, backward_vars_ptr, C_ptr, 
                                        grad_ptr, input_length, extended_label_length, num_labels);
}
'''

cpp_src = r'''
#include <torch/extension.h>
void forward_pass(float* network_outputs_ptr, int* extended_labels_ptr, float* forward_vars_ptr, float* C_ptr,
                int input_length, int extended_label_length, int num_labels);

void backward_pass(float* network_outputs_ptr, int* extended_labels_ptr, float* backward_vars_ptr, float* C_ptr,
                int input_length, int extended_label_length, int num_labels);

void cuda_grad(float* network_outputs_ptr, int* extended_labels_ptr, 
            float* forward_vars_ptr, float* backward_vars_ptr, float* C_ptr,
            float* grad_ptr, int input_length, int extended_label_length, int num_labels)
'''