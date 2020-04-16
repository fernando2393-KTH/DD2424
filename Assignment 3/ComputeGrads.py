"""
File with the computation of the gradients in numerical form.
It also includes an example of how to compute the difference with the analytical one.
"""

"""
def compute_grads_num_slow(data, labels, weights, bias, gamma, beta, lmb, h):
    grad_weights = list()
    grad_bias = list()
    grad_gamma = list()
    grad_beta = list()

    for j in range(len(bias)):
        grad_bias.append(np.zeros(len(bias[j])))
        for i in tqdm(range(len(bias[j]))):
            b_try = list()
            [b_try.append(np.copy(x)) for x in bias]
            b_try[j][i] = b_try[j][i] - h
            c1 = compute_cost(data, labels, weights, b_try, lmb, gamma, beta)
            b_try = list()
            [b_try.append(np.copy(x)) for x in bias]
            b_try[j][i] = b_try[j][i] + h
            c2 = compute_cost(data, labels, weights, b_try, lmb, gamma, beta)
            grad_bias[j][i] = (c2 - c1) / (2 * h)

    for j in range(len(weights)):
        grad_weights.append(np.zeros(weights[j].shape))
        for i in tqdm(range(grad_weights[-1].shape[0])):
            for k in range(grad_weights[-1].shape[1]):
                w_try = list()
                [w_try.append(np.copy(x)) for x in weights]
                w_try[j][i, k] = w_try[j][i, k] - h
                c1 = compute_cost(data, labels, w_try, bias, lmb, gamma, beta)
                w_try = list()
                [w_try.append(np.copy(x)) for x in weights]
                w_try[j][i, k] = w_try[j][i, k] + h
                c2 = compute_cost(data, labels, w_try, bias, lmb, gamma, beta)
                grad_weights[j][i, k] = (c2 - c1) / (2 * h)

    for j in range(len(gamma)):
        grad_gamma.append(np.zeros(gamma[j].shape))
        for i in tqdm(range(grad_gamma[-1].shape[0])):
            for k in range(grad_gamma[-1].shape[1]):
                g_try = list()
                [g_try.append(np.copy(x)) for x in gamma]
                g_try[j][i, k] = g_try[j][i, k] - h
                c1 = compute_cost(data, labels, weights, bias, lmb, g_try, beta)
                g_try = list()
                [g_try.append(np.copy(x)) for x in gamma]
                g_try[j][i, k] = g_try[j][i, k] + h
                c2 = compute_cost(data, labels, weights, bias, lmb, g_try, beta)
                grad_gamma[j][i, k] = (c2 - c1) / (2 * h)

    for j in range(len(beta)):
        grad_beta.append(np.zeros(beta[j].shape))
        for i in tqdm(range(grad_beta[-1].shape[0])):
            for k in range(grad_beta[-1].shape[1]):
                bt_try = list()
                [bt_try.append(np.copy(x)) for x in beta]
                bt_try[j][i, k] = bt_try[j][i, k] - h
                c1 = compute_cost(data, labels, weights, bias, lmb, gamma, bt_try)
                bt_try = list()
                [bt_try.append(np.copy(x)) for x in beta]
                bt_try[j][i, k] = bt_try[j][i, k] + h
                c2 = compute_cost(data, labels, weights, bias, lmb, gamma, bt_try)
                grad_beta[j][i, k] = (c2 - c1) / (2 * h)


    return grad_weights, grad_bias, grad_gamma, grad_beta
"""


"""
# Compare outcomes (averaged)
data, s_list, s_hat, mean_list, var_list = forward_pass(data_train, weights, bias, gamma, beta, None, None,
                                                        b_norm=True)
delta_w, delta_b, delta_g, delta_bt = compute_grads_analytic([i[:, 0:20] for i in data],
                                                             labels_train[:, 0:20],
                                                             weights, 0, data[-1][:, 0:20],
                                                             [i[:, 0:20] for i in s_list],
                                                             [i[:, 0:20] for i in s_hat],
                                                             gamma, mean_list, var_list, b_norm=True)

delta_w_num, delta_b_num, delta_g_num, delta_bt_num = compute_grads_num_slow(data_train[:, 0:20],
                                                                             labels_train[:, 0:20], weights,
                                                                             bias, gamma,
                                                                             beta, 0, 1e-5)
print([np.mean(abs(delta_b[i] - delta_b_num[i])) for i in range(len(delta_b))])
print([np.mean(abs(delta_w[i] - delta_w_num[i])) for i in range(len(delta_w))])
print([np.mean(abs(delta_g[i] - delta_g_num[i])) for i in range(len(delta_g))])
print([np.mean(abs(delta_bt[i] - delta_bt_num[i])) for i in range(len(delta_bt))])
"""
