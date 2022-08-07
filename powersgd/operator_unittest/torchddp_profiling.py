from end2end.torchddp import benchmark_xpu


if __name__ == '__main__':
    rank = 1
    print("rank = {}".format(low_rank))
    size = '500MB'
    iters = 10

    grad_size = int(float(size[:size.find('MB')]) * 262144)
    avg_e2e_time = benchmark_xpu(grad_size, iters, rank)
    print("size: {}, total: {:.2f}".format(size, avg_e2e_time))