"""
File with the computation of the gradients in numerical form.
It also includes an example of how to compute the difference with the analytical one.
"""

"""
def compute_grads_num(data, labels, weights, bias, lmb, h):
    grad_weights = list()
    grad_bias = list()

    c = compute_cost(data, labels, weights, bias, lmb)

    for j in range(len(bias)):
        grad_bias.append(np.zeros(len(bias[j])))
        for i in range(len(grad_bias[j])):
            b_try = list()
            [b_try.append(np.copy(x)) for x in bias]
            b_try[j][i] = b_try[j][i] + h
            c2 = compute_cost(data, labels, weights, b_try, lmb)
            grad_bias[j][i] = (c2 - c) / h

    for j in range(len(weights)):
        grad_weights.append(np.zeros(weights[j].shape))
        for i in tqdm(range(grad_weights[-1].shape[0])):
            for k in range(grad_weights[-1].shape[1]):
                w_try = list()
                [w_try.append(np.copy(x)) for x in weights]
                w_try[j][i, k] = w_try[j][i, k] + h
                c2 = compute_cost(data, labels, w_try, bias, lmb)
                grad_weights[j][i, k] = (c2 - c) / h

    return grad_weights, grad_bias
"""

"""
def compute_grads_num_slow(data, labels, weights, bias, lmb, h):
    grad_weights = list()
    grad_bias = list()

    for j in range(len(bias)):
        grad_bias.append(np.zeros(len(bias[j])))
        for i in range(len(bias[j])):
            b_try = list()
            [b_try.append(np.copy(x)) for x in bias]
            b_try[j][i] = b_try[j][i] - h
            c1 = compute_cost(data, labels, weights, b_try, lmb)
            b_try = list()
            [b_try.append(np.copy(x)) for x in bias]
            b_try[j][i] = b_try[j][i] + h
            c2 = compute_cost(data, labels, weights, b_try, lmb)
            grad_bias[j][i] = (c2 - c1) / (2 * h)

    for j in range(len(weights)):
        grad_weights.append(np.zeros(weights[j].shape))
        for i in tqdm(range(grad_weights[-1].shape[0])):
            for k in range(grad_weights[-1].shape[1]):
                w_try = list()
                [w_try.append(np.copy(x)) for x in weights]
                w_try[j][i, k] = w_try[j][i, k] - h
                c1 = compute_cost(data, labels, w_try, bias, lmb)
                w_try = list()
                [w_try.append(np.copy(x)) for x in weights]
                w_try[j][i, k] = w_try[j][i, k] + h
                c2 = compute_cost(data, labels, w_try, bias, lmb)
                grad_weights[j][i, k] = (c2 - c1) / (2 * h)

    return grad_weights, grad_bias
"""

"""
# Compare outcomes (averaged)
data, s_list = forward_pass(data_train, weights, bias)
delta_w, delta_b = compute_grads_analytic([i[:, 0:20] for i in data],
                                          labels_train[:, 0:20],
                                          weights, lmb, softmax(s_list[-1][:, 0:20]))
delta_w_num, delta_b_num = compute_grads_num_slow(data_train[:, 0:20], labels_train[:, 0:20], weights, bias, lmb, 1e-5)
print([abs(np.mean(delta_b[i] - delta_b_num[i])) for i in range(len(delta_b))])
print([abs(np.mean(delta_w[i] - delta_w_num[i])) for i in range(len(delta_w))])
"""
