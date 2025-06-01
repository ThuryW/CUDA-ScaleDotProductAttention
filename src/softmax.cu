#include "softmax.cuh"
#include <float.h> // For FLT_MAX
#include <math.h>  // For expf, fmaxf

__global__ void newSoftmaxKernel(const float* __restrict__ input, 
                                 float* __restrict__ output, 
                                 int /*rows_param*/, // This parameter is implicitly blockIdx.x * blockDim.x + threadIdx.x for more complex kernels, but here blockIdx.x is the row.
                                 int cols) {
    // Each block processes one row.
    int row_idx = blockIdx.x;

    // Thread ID within the block corresponds to the column index for this row.
    int col_idx = threadIdx.x;

    // Allocate shared memory for the current row's data.
    // blockDim.x should be equal to 'cols' for this kernel design.
    extern __shared__ float s_row_data[];

    // --- Step 1: Find the maximum value in the current row ---
    // Load data into shared memory.
    // Each thread loads one element of the row.
    if (col_idx < cols) {
        s_row_data[col_idx] = input[row_idx * cols + col_idx];
    } else {
        // In case blockDim.x > cols (not typical for this setup but for robustness)
        // pad with a value that won't affect the max.
        s_row_data[col_idx] = -FLT_MAX; 
    }
    __syncthreads(); // Ensure all data is loaded into shared memory.

    // Parallel reduction to find the maximum value in s_row_data.
    // Assumes blockDim.x is a power of 2 and equals cols.
    // If not, more complex handling for non-power-of-2 or cols > blockDim.x is needed.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (col_idx < s) {
            s_row_data[col_idx] = fmaxf(s_row_data[col_idx], s_row_data[col_idx + s]);
        }
        __syncthreads(); // Synchronize after each step of the reduction.
    }
    // The maximum value for the row is now in s_row_data[0].
    float max_val = s_row_data[0];
    // __syncthreads(); // Not strictly needed here as all threads will read same s_row_data[0]
                     // but good for clarity if s_row_data[0] was to be updated by thread 0 only.

    // --- Step 2: Subtract max, exponentiate ---
    // Each thread calculates exp(element - max_val) for its element.
    // This value will be used for summation and then for the final division.
    float exp_val_for_this_thread = 0.0f;
    if (col_idx < cols) {
        exp_val_for_this_thread = expf(input[row_idx * cols + col_idx] - max_val);
        s_row_data[col_idx] = exp_val_for_this_thread; // Store in shared memory for summation
    } else {
        s_row_data[col_idx] = 0.0f; // Padding threads contribute 0 to the sum.
    }
    __syncthreads(); // Ensure all exponentiated values are in shared memory.

    // --- Step 3: Sum the exponentiated values ---
    // Parallel reduction to sum the values in s_row_data.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (col_idx < s) {
            s_row_data[col_idx] += s_row_data[col_idx + s];
        }
        __syncthreads(); // Synchronize after each step of the reduction.
    }
    // The sum of exponentiated values for the row is now in s_row_data[0].
    float sum_exp = s_row_data[0];
    // __syncthreads(); // Again, not strictly needed here for reading s_row_data[0].

    // --- Step 4: Normalize and store the result ---
    // Each thread divides its exponentiated value by the sum.
    if (col_idx < cols) {
        if (sum_exp == 0.0f) { // Avoid division by zero (highly unlikely with expf unless all inputs were -inf).
            output[row_idx * cols + col_idx] = 0.0f; 
            // Alternatively, output[row_idx * cols + col_idx] = 1.0f / cols; for a uniform distribution.
        } else {
            // exp_val_for_this_thread holds the numerator calculated before sum reduction
            output[row_idx * cols + col_idx] = exp_val_for_this_thread / sum_exp;
        }
    }
}