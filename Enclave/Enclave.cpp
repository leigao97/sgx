#include "Enclave_t.h" /* print_string */
#include <sgx_trts.h>
#include <stdio.h> /* vsnprintf */
#include <string.h>
#include <algorithm>    // std::max


int printf(const char* fmt, ...)
{
    char buf[BUFSIZ] = { '\0' };
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, BUFSIZ, fmt, ap);
    va_end(ap);
    ocall_print_string(buf);
    return (int)strnlen(buf, BUFSIZ - 1) + 1;
}

//
float *r = nullptr;
float *precompute = nullptr;
void ecall_nativeMatMul(float* w, int* dimW, float* inp, int* dimInp, float* out){
    if (!w || !dimW || !inp || !dimInp || !out) {
        printf("Invalid input parameters\n");
        return;
    }

    int* dimW_ = new int [2];
    int* dimInp_ = new int [2];
    memcpy(dimW_, dimW, 2*sizeof(int));
    memcpy(dimInp_, dimInp, 2*sizeof(int));

    //inp_r and w_c retains
    int w_rows = dimW_[0];
    int inp_rows = dimInp_[0];
    int w_cols = dimW_[1];
    int inp_cols = dimInp_[1];
    //printf("%d %d %d %d", w_rows, w_cols, inp_rows, inp_cols);
    if(inp_cols != w_rows){
        printf("Invalid input parameters\n");
        return;
    }

    float* w_ = new float[w_rows * w_cols];
    float* inp_ = new float[inp_rows * inp_cols];

    
    
    memcpy(w_, w, w_rows * w_cols * sizeof(float));
    memcpy(inp_, inp, inp_rows * inp_cols * sizeof(float));

    float* out_ = new float[inp_rows * w_cols];
    //memset(out_, 0, sizeof(float) * inp_rows * w_cols);
    //addr: row*col_num + col
    for(int ir = 0; ir < inp_rows; ir++) for(int wc = 0; wc < w_cols; wc++){
        out_[ir*w_cols + wc] = 0;
        for(int ic = 0; ic < inp_cols; ic++)
            out_[ir*w_cols + wc] += inp_[ir*inp_cols + ic] * w_[wc*inp_cols + ic];    
    }
    /*
    for(int ir = 0; ir < batch; ir++) for(int wc = 0; wc < cols; wc++){
        ret[ir*cols + wc] = 0;
        for(int ic = 0; ic < rows; ic++){
            ret[ir*cols + wc] += r_[ir*rows + ic] * w_[wc*rows + ic];
        }
    }*/
    memcpy(out, out_, inp_rows * w_cols * sizeof(float));

    delete[] dimW_;
    delete[] dimInp_;
    delete[] w_;
    delete[] inp_;
    delete[] out_;
    
    return;
}

void ecall_precompute(float* weight, int* dim, int batch){
    if (!dim || !weight || !batch) {
        printf("Invalid input parameters\n");
        return;
    }
    
	int* dim_ = new int[2];
    memcpy(dim_, dim, 2*sizeof(int));
    int rows = dim[0];
    int cols = dim[1];
    float* w_ = new float[rows * cols];
    memcpy(w_, weight, rows * cols * sizeof(float));
    unsigned char* r_char = new unsigned char[batch * rows];
    float* ret;
    if(!r){
        ret = new float[batch * cols];
    }
    else{
        ret = precompute;
    }
    sgx_read_rand(r_char, batch * rows * sizeof(char));
    float* r_ = new float[batch * rows];
    for(int i = 0; i < batch * rows; i++) r_[i] = (float)r_char[i]/256;

    
    //addr: row*col_num + col
    for(int ir = 0; ir < batch; ir++) for(int wc = 0; wc < cols; wc++){
        ret[ir*cols + wc] = 0;
        for(int ic = 0; ic < rows; ic++){
            //printf("%f x %f\n", r_[ir*rows + ic], w_[wc*rows + ic]);
            ret[ir*cols + wc] += r_[ir*rows + ic] * w_[wc*rows + ic];
        }
    }
    r = r_;
    precompute = ret;
    delete[] dim_;
    delete[] w_;
    
    return;
}
//

void ecall_addNoise(float* inp, int* dim, float* out)
{
    
    // Check if the input dimensions are valid
    if (dim == nullptr || inp == nullptr || out == nullptr) {
        printf("Invalid input parameters\n");
        return;
    }

    // Initialize Dimension
	  int* dim_ = new int[2];
    memcpy(dim_, dim, 2*sizeof(int));
    int rows = dim[0];
    int cols = dim[1];

    /*
    // Allocate internal buffers
    float** inp_ = new float*[rows];
    float** out_ = new float*[rows];
    for (int i = 0; i < rows; i++) {
        inp_[i] = new float[cols];
        out_[i] = new float[cols];
    }
    */

    float* inp_ = new float[rows*cols];
    float* out_ = new float[rows*cols];

    // Copy the input data into the internal buffer
    memcpy(inp_, inp, rows * cols * sizeof(float));

    // Check if the global variable 'r' is initialized
    if (r == nullptr) {
        printf("Global variable 'r' is not initialized\n");
        // Free allocated memory
        /*
        for (int i = 0; i < rows; i++) {
            delete[] inp_[i];
            delete[] out_[i];
        }
        */
        delete[] inp_;
        delete[] out_;
        return;
    }
    // Perform matrix addition inp + r
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out_[i * cols + j] = inp_[i * cols + j] + r[i * cols + j];
        }
    }
    // Copy the results back to the untrusted memory
    memcpy(out, out_, rows * cols * sizeof(float));

    // Free allocated memory
    /*
    for (int i = 0; i < rows; i++) {
        delete[] inp_[i];
        delete[] out_[i];
    }
    */
    delete[] inp_;
    delete[] out_;
    
    return;
}

void ecall_removeNoise(float* inp, int* dim, float* out)
{
    // Check if the input dimensions are valid
    if (dim == nullptr || inp == nullptr || out == nullptr) {
        printf("Invalid input parameters\n");
        return;
    }

    // Initialize Dimension
	  int* dim_ = new int[2];
    memcpy(dim_, dim, 2*sizeof(int));
    int rows = dim[0];
    int cols = dim[1];

    /*
    // Allocate internal buffers
    float** inp_ = new float*[rows];
    float** out_ = new float*[rows];
    for (int i = 0; i < rows; i++) {
        inp_[i] = new float[cols];
        out_[i] = new float[cols];
    }
    */
    float* inp_ = new float[rows*cols];
    float* out_ = new float[rows*cols];

    // Copy the input data into the internal buffer
    memcpy(inp_, inp, rows * cols * sizeof(float));

    // Check if the global variable 'precompute' is initialized
    if (precompute == nullptr) {
        printf("Global variable 'precompute' is not initialized\n");
        // Free allocated memory
        /*
        for (int i = 0; i < rows; i++) {
            delete[] inp_[i];
            delete[] out_[i];
        }
        */
        delete[] inp_;
        delete[] out_;
        return;
    }

    // Perform matrix subtraction inp - precompute
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out_[i*cols+j] = inp_[i*cols+j] - precompute[i*cols+j];
        }
    }

    // Copy the results back to the untrusted memory
    memcpy(out, out_, rows * cols * sizeof(float));

    // Free allocated memory
    /*
    for (int i = 0; i < rows; i++) {
        delete[] inp_[i];
        delete[] out_[i];
    }
    */
    delete[] inp_;
    delete[] out_;
    return;
}


// the actual buffer of *inp is in untrusted memory
// You can read from it, but never write to it
int ecall_compute_secrete_operation(int* inp, int size) {
    // decrypt inp
    // ....

    int res = 0;

    for (int i = 0; i < size; i++) {
        res += inp[i];
    }

    // encrypt res
    // ....

    printf("Returning to App.cpp\n");
    return res;
}
