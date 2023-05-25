#include <stdio.h>

// Kernel function for vector addition
_global_ void vectorAdd(int *a, int *b, int *c, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
    {
        c[tid] = a[tid] + b[tid];
    }
}

int main()
{
    int size = 1000000;   // Size of the vectors
    int *h_a, *h_b, *h_c; // Host vectors
    int *d_a, *d_b, *d_c; // Device vectors

    // Allocate memory for host vectors
    h_a = (int *)malloc(size * sizeof(int));
    h_b = (int *)malloc(size * sizeof(int));
    h_c = (int *)malloc(size * sizeof(int));

    // Initialize host vectors with values
    for (int i = 0; i < size; i++)
    {
        h_a[i] = i;
        h_b[i] = i;
    }

    // Allocate memory for device vectors
    cudaMalloc((void **)&d_a, size * sizeof(int));
    cudaMalloc((void **)&d_b, size * sizeof(int));
    cudaMalloc((void **)&d_c, size * sizeof(int));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(int), cudaMemcpyHostToDevice);

    // Perform vector addition on device
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the first few elements of the result
    for (int i = 0; i < 10; i++)
    {
        printf("%d ", h_c[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}