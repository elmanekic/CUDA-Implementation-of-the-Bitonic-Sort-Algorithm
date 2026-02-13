#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <climits>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>


#define BLOCK_SIZE 1024

static void cudaErrorCheck(const char* msg, cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}

// Kernel koji sortira jedan blok koristeći shared memoriju
__global__ void bitonicSharedKernel(int *data, int n) {
    __shared__ int s_data[BLOCK_SIZE];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;

    if (gid < n)
        s_data[tid] = data[gid];
    else
        s_data[tid] = INT_MAX;

    __syncthreads();

    for (int k = 2; k <= blockDim.x; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int ixj = tid ^ j;

            if (ixj > tid) {
                bool ascending = ((tid & k) == 0);
                int val_tid = s_data[tid];
                int val_ixj = s_data[ixj];

                if ((ascending && val_tid > val_ixj) ||
                    (!ascending && val_tid < val_ixj)) {
                    s_data[tid] = val_ixj;
                    s_data[ixj] = val_tid;
                }
            }
            __syncthreads();
        }
    }

    if (gid < n)
        data[gid] = s_data[tid];
}

// Kernel za globalni bitonic merge
__global__ void bitonicGlobalMerge(int *data, int n, int j, int k) {
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= n) return;

    unsigned int ixj = i ^ j;
    if (ixj > i && ixj < n) {
        bool ascending = ((i & k) == 0);
        int valI = data[i];
        int valIXJ = data[ixj];

        if ((ascending && valI > valIXJ) || (!ascending && valI < valIXJ)) {
            data[i] = valIXJ;
            data[ixj] = valI;
        }
    }
}

void bitonicSortShared(int *deviceData, int n) {
    int threads = BLOCK_SIZE;
    int blocks = (n + threads - 1) / threads;
    cudaError_t err;

    bitonicSharedKernel<<<blocks, threads>>>(deviceData, n);
    err = cudaDeviceSynchronize();
    cudaErrorCheck("bitonicSharedKernel failed", err);

    for (int k = BLOCK_SIZE; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonicGlobalMerge<<<blocks, threads>>>(deviceData, n, j, k);
            err = cudaDeviceSynchronize();
            cudaErrorCheck("bitonicGlobalMerge failed", err);
        }
    }
}

int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}


int main() {
    const int n = 50000000;
    int paddedN = nextPow2(n);


    size_t size = paddedN * sizeof(int);

    
    int *h_data = (int*)malloc(size);
    for (int i = 0; i < n; ++i)
        h_data[i] = rand() % 100000;

    for (int i = n; i < paddedN; ++i)
        h_data[i] = INT_MAX;

    int *h_backup = (int*)malloc(n * sizeof(int));
    memcpy(h_backup, h_data, n * sizeof(int));

    int *d_data;
    cudaError_t err;

    err = cudaMalloc(&d_data, size);
    cudaErrorCheck("cudaMalloc(d_data)", err);

    err = cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);
    cudaErrorCheck("cudaMemcpy H2D", err);


    // MJERENJE VREMENA
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bitonicSortShared(d_data, paddedN);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("GPU sort time: %.3f ms\n", elapsedTime);

    

    err = cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    cudaErrorCheck("cudaMemcpy D2H", err);

 

    if (std::is_sorted(h_data, h_data + n))
        printf("Sorted!\n");
    else
        printf("Sort FAILED!\n");

    cudaFree(d_data);
    


 //---Radix sort---
    thrust::device_vector<int> d_radix(h_backup, h_backup + n);

    cudaEventRecord(start);
    thrust::sort(d_radix.begin(), d_radix.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float radix_ms;
    cudaEventElapsedTime(&radix_ms, start, stop);

    printf("Thrust RADIX sort time: %.3f ms\n", radix_ms);

    //---Merge sort---

        thrust::device_vector<int> d_merge(h_backup, h_backup + n);

    cudaEventRecord(start);
    thrust::stable_sort(d_merge.begin(), d_merge.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float merge_ms;
    cudaEventElapsedTime(&merge_ms, start, stop);

    printf("Thrust MERGE sort time: %.3f ms\n", merge_ms);


    printf("========== SUMMARY ==========\n");
    printf("Bitonic: %.3f ms\n", elapsedTime);
    printf("Radix  : %.3f ms\n", radix_ms);
    printf("Merge  : %.3f ms\n", merge_ms);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_data);
    free(h_backup);



    return 0;
}
