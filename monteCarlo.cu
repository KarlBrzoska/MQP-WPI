#include <stdio.h>
#include <curand_kernel.h>
#include <ctime>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 2048
#define NUM_SAMPLES 1000000000000

__global__ void initRandomStates(curandState *states, unsigned long long seed) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

__global__ void monteCarloPiKernel(unsigned long long *count, curandState *states) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned long long local_count = 0;
    curandState localState = states[idx];

    // Each thread will do an equal amount of work
    unsigned long long iterations = NUM_SAMPLES / (blockDim.x * gridDim.x);

    for (unsigned long long i = 0; i < iterations; ++i) {
        float x = curand_uniform(&localState);
        float y = curand_uniform(&localState);
        if (x * x + y * y <= 1.0f) {
            local_count++;
        }
    }

    // Copy state back to global memory
    states[idx] = localState;

    // Use atomicAdd to avoid race condition
    atomicAdd(count, local_count);
}

int main() {
    // Start CPU timing
    clock_t start_cpu = clock();

    unsigned long long *d_count;
    curandState *d_states;

    // Allocate memory on the device
    cudaMalloc(&d_count, sizeof(unsigned long long));
    cudaMalloc(&d_states, NUM_BLOCKS * THREADS_PER_BLOCK * sizeof(curandState));

    // Initialize d_count to 0
    cudaMemset(d_count, 0, sizeof(unsigned long long));

    // Setup PRNG states
    initRandomStates<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_states, 1337ULL);

    // Wait for GPU to finish before launching the main kernel
    cudaDeviceSynchronize();

    // Create CUDA events for timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // Record the start event for GPU timing
    cudaEventRecord(start_gpu, NULL);

    // Run kernel
    monteCarloPiKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_count, d_states);

    // Record the stop event for GPU timing
    cudaEventRecord(stop_gpu, NULL);
    cudaEventSynchronize(stop_gpu);

    // Calculate the elapsed time for GPU operations
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    // Copy count back to host
    unsigned long long h_count;
    cudaMemcpy(&h_count, d_count, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Calculate pi
    double pi = 4.0 * h_count / NUM_SAMPLES;

    // End CPU timing
    clock_t end_cpu = clock();
    double cpu_time_used = ((double) (end_cpu - start_cpu)) / CLOCKS_PER_SEC;

    printf("Estimated pi: %f\n", pi);
    printf("GPU time elapsed: %f milliseconds\n", milliseconds);
    printf("Total CPU time used: %f seconds\n", cpu_time_used);

    // Cleanup
    cudaFree(d_count);
    cudaFree(d_states);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}
