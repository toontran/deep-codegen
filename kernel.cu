#include "kernel.h"
#include <math.h>

__global__ void vector_add(float *A, float *B, float *C, int n, int U){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int LB = tid * U;
    int UB = min((tid+1)*U, n);
    for (int i=LB; i<UB; i++) {
        //printf("A[%d]=%.3f\n", i, A[i]);
        C[i] = A[i] + B[i];
    }
}

__global__ void matmul(float *A, float *B, float *C, int n, int m, int p) {
//     for (int i=0; i<m*n; i++) 
//         printf("input1[%d] = %.3lf, input2[%d] = %.3lf\n", i, A[i], i, B[i]);
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid1 = tid / p;
    int tid2 = tid % p;
    
    if (tid1 > n) return;
    
    for (int i=0; i<m; i++) {
        //printf("C[%d] +=A[%d] * B[%d],   %.3lf += %.3lf * %.3lf\n", tid1*p+tid2, tid1*m + i, i*p + tid2, C[tid1*p+tid2], A[tid1*m + i], B[i*p + tid2]);
        C[tid1*p+tid2] += A[tid1*m + i] * B[i*p + tid2];
    }
}

// __global__ void vector_add(float *A, float *B, float *C, int n, int U){
//     print(A)
// }

void sum_two_tensors(array1d_t<float>& a, array1d_t<float>& b, array1d_t<float>& output){
    int W = a.col_count;
    int numBlocks = 10;
    int numThreadsPerBlock = 10;
    int U = ceil((float) W / (numBlocks*numThreadsPerBlock));
//     printf("W: %d, U: %d\n", W, U);
    vector_add<<<numBlocks, numThreadsPerBlock>>>(a.data_ptr, b.data_ptr, output.data_ptr, W, U);
    cudaDeviceSynchronize();
}

// Need to try output 2d matrix here, then implement matrix multiplication
void gemm(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output){
    int n = input1.row_count;
    int m = input1.col_count;
    int p = input2.col_count;
    int o1 = output.row_count;
    int o2 = output.col_count;
    //printf("shapes: (%d, %d) @ (%d, %d) = (%d, %d)\n", n, m, input2.row_count, p, o1, o2);
    //printf("input1[0][0], input1[%d][%d] = %d\n", input1.row_count, input1.col_count, input1.data_ptr[0][0]);
    //printf("input1[0][0], input1[%d][%d] = %.3f\n", input1.row_count, input1.col_count);
    int W = n * p;
    float numWarps = (float) W / 32;
    int numBlocks = ceil(numWarps / 8);
    //printf("numBlocks: %d, %.3f, total work: %d, num threads/block: %d\n", numBlocks, numWarps, W, W/numBlocks+1);
    matmul<<<numBlocks, W/numBlocks+1>>>(input1.data_ptr, input2.data_ptr, output.data_ptr, n, m, p);
}

// Then the bias later

void gspmmv(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& output, bool reverse, bool norm){;}
void gspmmve(graph_t& graph, array2d_t<float>& input1, array1d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmme(graph_t& graph, array1d_t<float>& edge_input, array1d_t<float>& output, op_t op, bool reverse){;}
void gspmme2d(graph_t& graph, array2d_t<float>& edge_input, array2d_t<float>& output, op_t op, bool reverse){;}
void gspmmve2d(graph_t& graph, array3d_t<float>& input1, array2d_t<float>& edge_input, array3d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve(graph_t& graph, array1d_t<float>& input_left, array1d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmve2d(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv(graph_t& graph, array2d_t<float>& input_left, array2d_t<float>& input_right, array1d_t<float>& output, op_t op, bool reverse){;}
void gsddmmvv2d(graph_t& graph, array3d_t<float>& input_left, array3d_t<float>& input_right, array2d_t<float>& output, op_t op, bool reverse){;}
void test_2out(graph_t& graph, array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test3(array2d_t<float>& input1, array2d_t<float>& input2, array2d_t<float>& output1, array2d_t<float>& output2, op_t op, bool reverse){;}
void test4(array3d_t<float>& input1, array4d_t<float>& input2, array4d_t<float>& output1, int t){;}
