__kernel void matmul(
    __global const float* A,
    __global const float* B,
    __global float* C,
    int M, int N, int K
) {
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= M || col >= N)
        return;

    float acc = 0.0f;

    int a_base = row * K;
    int b_base = col;

    for (int k = 0; k < K; k++) {
        acc += A[a_base + k] * B[k * N + b_base];
    }

    C[row * N + col] = acc;
}
