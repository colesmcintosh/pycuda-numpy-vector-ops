# pycuda-numpy-vector-ops
This notebook demonstrates how to accelerate large-scale NumPy operations using GPU programming in Python via [PyCUDA](https://documen.tician.de/pycuda/).

We compare traditional CPU-based NumPy operations with a GPU-accelerated fused multiply-add (FMA) operation:

> The operation is defined as $c[i] = a[i] \times b[i] + d[i]$.

The notebook uses:
- Pinned (page-locked) memory for faster host-device transfers
- CUDA streams for asynchronous execution
- Event timing for accurate benchmarks

The result is a fast, validated comparison of NumPy vs PyCUDA performance.
