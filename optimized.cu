#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <climits>
#include <cstring>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define BLOCK_SIZE 1024  // broj threadova po bloku
#define TILE_SIZE 1024*4 // koliko elemenata ide u shared memory po bloku (može se povećati)

__device__ inline int compare(int a, int b, bool dir) {
    return dir ? min(a,b) : max(a,b);
}

// ----- Shared memory bitonic sort za blok -----

__global__ void bitonicSharedKernel(int* d_data, int n) {

extern __shared__ int s_data[];


unsigned int tid = threadIdx.x;
unsigned int gid = blockIdx.x * blockDim.x + tid;

// Učitaj elemente u shared memory

if (gid < n)
s_data[tid] = d_data[gid];

__syncthreads();



// Bitonic sort unutar shared memory

for (unsigned int size = 2; size <= blockDim.x; size <<= 1) {
    for (unsigned int stride = size >> 1; stride > 0; stride >>= 1) {
        unsigned int pos = 2 * tid - (tid & (stride - 1));

if (pos + stride < blockDim.x) {
    bool dir = ((pos & size) == 0);
    int val = compare(s_data[pos], s_data[pos + stride], dir);
    int val2 = compare(s_data[pos + stride], s_data[pos], dir);
    s_data[pos] = val;
    s_data[pos + stride] = val2;

}

__syncthreads();

}
}
// Vrati rezultat u global memory

if (gid < n)

d_data[gid] = s_data[tid];

}


// ----- Shared memory bitonic sort za blok -----
__global__ void bitonicSharedMultiStep(int* d_data, int n, int k) {
    extern __shared__ int s_data[];

    unsigned int tid = threadIdx.x;
    // Svaki blok obrađuje BLOCK_SIZE * 2 elemenata (svaki thread uzima 2)
    unsigned int offset = blockIdx.x * (blockDim.x * 2);
    
    // Učitavanje parova u shared memory
    // Koristimo istu logiku indeksiranja kao u optimiziranom globalnom mergeu
    unsigned int l = tid & (BLOCK_SIZE - 1);
    unsigned int h = (tid & ~(BLOCK_SIZE - 1)) << 1;
    unsigned int idx1 = h | l;
    unsigned int idx2 = idx1 | BLOCK_SIZE;

    if (offset + idx2 < n) {
        s_data[idx1] = d_data[offset + idx1];
        s_data[idx2] = d_data[offset + idx2];
    }
    __syncthreads();

    // Unutar shared memorije rješavamo SVE korake j od k/2 ili BLOCK_SIZE naniže
    for (int j = BLOCK_SIZE; j > 0; j >>= 1) {
        unsigned int sl = tid & (j - 1);
        unsigned int sh = (tid & ~(j - 1)) << 1;
        unsigned int s_idx1 = sh | sl;
        unsigned int s_idx2 = s_idx1 | j;

        bool dir = ((offset + s_idx1) & k) == 0;
        
        if ((s_data[s_idx1] > s_data[s_idx2]) == dir) {
            int tmp = s_data[s_idx1];
            s_data[s_idx1] = s_data[s_idx2];
            s_data[s_idx2] = tmp;
        }
        __syncthreads();
    }

    // Vraćanje u globalnu memoriju
    if (offset + idx2 < n) {
        d_data[offset + idx1] = s_data[idx1];
        d_data[offset + idx2] = s_data[idx2];
    }
}

__global__ void bitonicGlobalMergeOptimized(int* d_data, int n, int j, int k) {
    // Svaki thread obrađuje jedan PAR elemenata
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Izračun indeksa parova pomoću bitwise operacija
    unsigned int l = i & (j - 1);
    unsigned int h = (i & ~(j - 1)) << 1;
    unsigned int idx1 = h | l;
    unsigned int idx2 = idx1 | j;

    if (idx2 < n) {
        int val1 = d_data[idx1];
        int val2 = d_data[idx2];
        
        // Smjer sortiranja (ascending/descending)
        bool dir = (idx1 & k) == 0;
        
        if ((val1 > val2) == dir) {
            d_data[idx1] = val2;
            d_data[idx2] = val1;
        }
    }
}

// ----- Nova helper funkcija -----
void bitonicSort(int* d_data, int n) {
    int sharedMemSize = (BLOCK_SIZE * 2) * sizeof(int);
    int numBlocksMerge = (n / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // 1. Inicijalni sort (Shared) ostaje isti
    bitonicSharedKernel<<<n / BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_data, n);

    // 2. Glavna petlja
    for (int k = BLOCK_SIZE * 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            if (j >= BLOCK_SIZE) {
                // Ako je razmak veći od bloka, moramo koristiti globalni pristup
                bitonicGlobalMergeOptimized<<<numBlocksMerge, BLOCK_SIZE>>>(d_data, n, j, k);
            } else {
                // Ako j padne ispod BLOCK_SIZE, jedan kernel rješava SVE preostale j-ove
                bitonicSharedMultiStep<<<numBlocksMerge, BLOCK_SIZE, sharedMemSize>>>(d_data, n, k);
                break; // Izlazimo iz unutrašnje j petlje jer je MultiStep odradio j=512...1
            }
        }
    }
    cudaDeviceSynchronize();
}

// Provjera da li je n potencija broja 2
bool isPowerOfTwo(int n) {
    return (n > 0) && ((n & (n - 1)) == 0);
}

// Najmanja potencija broja 2 koja je >= n
int nextPowerOfTwo(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}


int main() {

    int original_n = 50000000;   
    int n = original_n;

    if (!isPowerOfTwo(n)) {
        int new_n = nextPowerOfTwo(n);
        std::cout << "Padding from " << n << " to " << new_n << std::endl;
        n = new_n;
    }

    size_t size = n * sizeof(int);

    // CPU data
    int* h_data = new int[n];
    int* h_backup = new int[original_n];

    // Popuni originalne vrijednosti
    for (int i = 0; i < original_n; i++)
        h_data[i] = rand();

    memcpy(h_backup, h_data, original_n * sizeof(int));

    // Padding sa INT_MAX
    for (int i = original_n; i < n; i++)
        h_data[i] = INT_MAX;

    // GPU data
    int* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // GPU events za mjerenje vremena
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Bitonic sort
    cudaEventRecord(start); // start timer
    bitonicSort(d_data, n);
    cudaEventRecord(stop);  // stop timer
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("GPU bitonic sort time: %.3f ms\n", milliseconds);

    // Copy back
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    // Validate
    bool ok = true;
    for (int i = 1; i < original_n; i++)
        if (h_data[i-1] > h_data[i]) { ok = false; break; }

    printf("Sorted correctly? %s\n", ok ? "YES" : "NO");

    // Cleanup
    cudaFree(d_data);
    delete[] h_data;

    //Radix sort
    thrust::device_vector<int> d_radix(h_backup, h_backup + original_n);
    cudaEventRecord(start);
    thrust::sort(d_radix.begin(), d_radix.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float radix_ms;
    cudaEventElapsedTime(&radix_ms, start, stop);
    printf("Thrust RADIX sort time: %.3f ms\n", radix_ms);

    //Merge sort
    thrust::device_vector<int> d_merge(h_backup, h_backup + original_n);
    cudaEventRecord(start);
    thrust::stable_sort(d_merge.begin(), d_merge.end());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float merge_ms;
    cudaEventElapsedTime(&merge_ms, start, stop);
    printf("Thrust MERGE sort time: %.3f ms\n", merge_ms);



    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(h_data);
    free(h_backup);

    return 0;
}
