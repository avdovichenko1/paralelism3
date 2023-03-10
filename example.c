#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define N 10000000
#define ALPHA 1.0f

int main() {
    float *arr, *d_arr;
    float Sum = 0;
    clock_t a = clock();

    // Allocate memory for the arrays
    arr = (float*)malloc(N * sizeof(float));
    cudaMalloc(&d_arr, N * sizeof(float));

    // Initialize the array on the CPU
    for (int i = 0; i < N; i++)
        arr[i] = sinf(2 * M_PI * i / N);

    // Copy the array to the GPU
    cudaMemcpy(d_arr, arr, N * sizeof(float), cudaMemcpyHostToDevice);

    // Create a handle for the cuBLAS library
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Perform the vector operation
    cublasSaxpy(handle, N, &ALPHA, d_arr, 1, d_arr, 1);

    // Copy the result back to the CPU
    cudaMemcpy(arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Wait for all CUDA work to complete
    cudaDeviceSynchronize();

    // Sum the array on the CPU
    for (int i = 0; i < N; i++)
        Sum += arr[i];

    // Print the result
    printf("%.25f\n", Sum);

    // Destroy the cuBLAS handle
    cublasDestroy(handle);

    // Free the memory
    free(arr);
    cudaFree(d_arr);

    clock_t b = clock();
    double d = (double)(b-a) / CLOCKS_PER_SEC;
    printf("%.25f\n", d);

    return 0;
}

