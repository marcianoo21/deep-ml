#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

__global__ void index_kernel(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx;
    }
}

std::vector<int> global_thread_indices(int n) {
    if (n <= 0) return {};

    int* d_out = nullptr;
    size_t bytes = static_cast<size_t>(n) * sizeof(int);

    cudaError_t err = cudaMalloc(&d_out, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
    }

    // 2. launch kernel with enough threads to cover n
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock; // ceil division

    index_kernel<<<blocks, threadsPerBlock>>>(d_out, n);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_out);
        throw std::runtime_error(std::string("Kernel launch failed: ") + cudaGetErrorString(err));
    }

    // 3. copy result back to host
    std::vector<int> result(n);
    err = cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_out);

    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy failed: ") + cudaGetErrorString(err));
    }

    return result;
}