from end2end.hipress_mxnet import benchmark_xpu
# import ..hipress_mxnet
# import ..hipress_mxnet.benchmark_xpu


if __name__ == '__main__':
    rank = 1
    sizes = ['500MB']
    iters = 10
    print("rank = {}".format(rank))
    for size in sizes:
        # 1 MB = 1048576 Bytes = 262144 float32s
        grad_size = int(float(size[:size.find('MB')]) * 262144)
        encode1, encode2, decode, end2end = (1000 * i for i in benchmark_xpu(grad_size, iters, rank))
        print("size: {}, total: {:.2f}, encode1: {:.2f}, encode2: {:.2f}, decode: {:.2f}".format(
            size,
            end2end,
            encode1,
            encode2,
            decode
        ))