"""
import numpy as np
import Assignment4
import copy

def computeGrads(rnn, x, y, h):
    res_rnn = copy.deepcopy(rnn)
    h_prev = np.zeros(rnn.m)
    for idx, att in enumerate(['b', 'c', 'u', 'w', 'v']):
        grad = np.zeros(getattr(rnn, att).shape)
        for i in range(grad.shape[0]):
            for j in range(grad.shape[1]):
                rnn_try = copy.deepcopy(rnn)
                aux = np.copy(getattr(rnn_try, att))
                aux[i, j] -= h
                setattr(rnn_try, att, aux)
                p = forward(rnn_try, h_prev, x)[0]
                l1 = compute_loss(y, p)
                rnn_try = copy.deepcopy(rnn)
                aux = np.copy(getattr(rnn_try, att))
                aux[i, j] += h
                setattr(rnn_try, att, aux)
                p = forward(rnn_try, h_prev, x)[0]
                l2 = compute_loss(y, p)
                grad[i, j] = (l2 - l1) / (2 * h)
        setattr(res_rnn, att, grad)

    return res_rnn


rnn_test = computeGrads(rnn, x, y, 1e-4)
print(np.mean(abs(rnn_grads.b - rnn_test.b)))
print(np.mean(abs(rnn_grads.c - rnn_test.c)))
print(np.mean(abs(rnn_grads.u - rnn_test.u)))
print(np.mean(abs(rnn_grads.v - rnn_test.v)))
print(np.mean(abs(rnn_grads.w - rnn_test.w)))
"""
