"""
This is the translation of the matlab computations to python
"""


"""
def compute_grads_num(data, labels, weight, bias, h, lmb):
    grad_weight = np.zeros(weight.shape)
    grad_bias = np.zeros(bias.shape)

    c = compute_cost(data, labels, weight, bias, lmb)

    for i in range(bias.shape[0]):
        bias_aux = np.copy(bias)
        bias_aux[i] += h
        c2 = compute_cost(data, labels, weight, bias_aux, lmb)
        grad_bias[i] = (c2 - c) / h

    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            weight_aux = np.copy(weight)
            weight_aux[i, j] += h
            c2 = compute_cost(data, labels, weight_aux, bias, lmb)
            grad_weight[i, j] = (c2 - c) / h

    return grad_weight, grad_bias
"""

"""
def compute_grads_num_slow(data, labels, weight, bias, h, lmb):
    grad_weight = np.zeros(weight.shape)
    grad_bias = np.zeros(bias.shape)

    for i in range(bias.shape[0]):
        bias_aux = np.copy(bias)
        bias_aux[i] -= h
        c1 = compute_cost(data, labels, weight, bias_aux, lmb)
        bias_aux = np.copy(bias)
        bias_aux[i] += h
        c2 = compute_cost(data, labels, weight, bias_aux, lmb)
        grad_bias[i] = (c2 - c1) / (2 * h)

    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            weight_aux = np.copy(weight)
            weight_aux[i, j] -= h
            c1 = compute_cost(data, labels, weight_aux, bias, lmb)
            weight_aux = np.copy(weight)
            weight_aux[i, j] += h
            c2 = compute_cost(data, labels, weight_aux, bias, lmb)
            grad_weight[i, j] = (c2 - c1) / (2 * h)

    return grad_weight, grad_bias
"""