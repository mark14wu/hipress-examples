void powersgd_encode1_body(float* grad, float* residual, float* Q, float* M, float* P,) {
    int N = grad.size;
    M[:N] = grad.flatten() + residual;
    hipress::matmul(M, Q, out=P).copy_(P);
}
void powersgd_encode2_body(float* P, float* M, float* Q) {
    hipress::orthogonalize(P);
    hipress::matmul(M.t(), P, out=Q);
}
void powersgd_decode_body(float* P, float* Q, float* M, float* residual, float* grad) {
    int64_t N = grad.size;
    hipress::matmul(P, Q.t(), out=M);
    residual = grad - M[:N];
    grad = M[:N];
}