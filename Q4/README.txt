This contains two implementations of Connectionist Temporal Classification (CTC):

1. Picovoice_q4.py
   - Python-only version of the CTC implementation
   - Pure Python implementation without GPU acceleration

2. Picovoice_q4_cuda.py
   - Python/CUDA version of the CTC implementation
   - Utilizes cuda_utils.py for parallelized and optimized CTC operations