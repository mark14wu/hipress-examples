namespace hipress {
void matmul(float* a, float* b, float* out) {
    torch::mm(a, b).copy_(out);
}
void orthogonalize(float* P) {
    torch::Tensor p1, p2;
    std::tie(p1, p2) = torch::geqrf(P);
    torch::orgqr(p1, p2).copy_(P);
}
}