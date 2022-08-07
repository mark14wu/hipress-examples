import mxnet as mx
import time
import math


def benchmark_xpu(grad_size, iters, rank):
    ctx = mx.context.gpu(3)

    grad_random = mx.nd.normal(shape=grad_size, ctx=ctx, dtype='float32')
    grad_zeros = mx.nd.zeros(shape=grad_size, ctx=ctx, dtype='float32')
    grad = mx.nd.zeros(shape=grad_size, ctx=ctx, dtype='float32')
    n = math.ceil(math.sqrt(grad.size))
    res = mx.nd.zeros_like(grad)

    p = mx.nd.zeros(shape=(n, rank), ctx=ctx, dtype=grad.dtype)
    q = mx.nd.normal(shape=(n, rank), ctx=ctx, dtype=grad.dtype)
    m = mx.nd.zeros(shape=(n, n), ctx=ctx, dtype=grad.dtype)
    
    encode1_time = []
    encode2_time = []
    decode_time = []
    e2e_time = []

    for i in range(iters):
        grad_random.copyto(grad)
        grad.wait_to_read()

        t_start = time.time()
        mx.nd.contrib.power_sgd_encode1(
            grad=grad,
            q=q,
            residual=res,
            m=m,
            out=p
        )
        mx.nd.contrib.power_sgd_encode2(
            p=p,
            m=m,
            out=q
        )
        mx.nd.contrib.power_sgd_decode(
        grad=grad,
        q=q,
        residual=res,
        m=m,
        p=p
        )
        m.wait_to_read()
        res.wait_to_read()
        grad.wait_to_read()
        q.wait_to_read()
        p.wait_to_read()
        t_end = time.time()

        e2e_time.append(t_end - t_start)
    
    for i in range(iters):
        grad_random.copyto(grad)
        # grad_zeros.copyto(grad)
        grad.wait_to_read()

        # print("encode1...")
        t1_start = time.time()
        mx.nd.contrib.power_sgd_encode1(
            grad=grad,
            q=q,
            residual=res,
            m=m,
            out=p
        )
        grad.wait_to_read()
        q.wait_to_read()
        res.wait_to_read()
        m.wait_to_read()
        p.wait_to_read()
        t1_end = time.time()

        # print("encode2...")
        t2_start = time.time()
        mx.nd.contrib.power_sgd_encode2(
            p=p,
            m=m,
            out=q
        )
        p.wait_to_read()
        m.wait_to_read()
        q.wait_to_read()
        t2_end = time.time()

        # print("decode...")
        t3_start = time.time()
        mx.nd.contrib.power_sgd_decode(
        grad=grad,
        q=q,
        residual=res,
        m=m,
        p=p
        )
        m.wait_to_read()
        res.wait_to_read()
        grad.wait_to_read()
        q.wait_to_read()
        p.wait_to_read()
        t3_end = time.time()

        encode1_time.append(t1_end - t1_start)
        encode2_time.append(t2_end - t2_start)
        decode_time.append(t3_end - t3_start)
    encode1_time.sort()
    encode2_time.sort()
    decode_time.sort()
    e2e_time.sort()
    head = int(iters * 0.25)
    tail = int(iters * 0.75)
    encode1_time = encode1_time[head:tail]
    encode2_time = encode2_time[head:tail]
    decode_time = decode_time[head:tail]
    e2e_time = e2e_time[head:tail]
    avg_encode1 = sum(encode1_time) / len(encode1_time)
    avg_encode2 = sum(encode2_time) / len(encode2_time)
    avg_decode = sum(decode_time) / len(decode_time)
    avg_e2e = sum(e2e_time) / len(e2e_time)
    return avg_encode1, avg_encode2, avg_decode, avg_e2e

if __name__ == '__main__':
    rank = 1
    sizes = ['0.01MB', '0.1MB', '1MB', '10MB', '100MB', '500MB']
    iters = 200
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