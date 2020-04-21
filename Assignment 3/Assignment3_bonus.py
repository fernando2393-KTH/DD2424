import numpy as np
import matplotlib.pyplot as plt
import os.path
from tqdm import tqdm

DATAPATH = '../Datasets/cifar-10-batches-py/'
LENGTH = 1024  # Number of pixels of the image
SIZE = 32  # Pixel dimension of the image
D_BATCH = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
T_BATCH = 'test_batch'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='latin1')
    return dictionary


def find_best_n(array, n):
    array_copy = np.copy(array)
    indices = list()  # Queue of ordered indices of the n maximum elements
    for i in range(n):
        index = np.argmax(array_copy)  # Get index of maximum
        array_copy[index] = -1  # Set maximum element to -1 (all are > 0)
        indices.append(int(index))  # Insert index as a list in the queue

    return indices


def read_data(size_val=5000):
    file = unpickle(DATAPATH + D_BATCH[0])
    data_train = file['data']  # Images data for training
    labels_train = file['labels']  # Images labels for training
    for i in range(1, len(D_BATCH)):  # Compose the training set
        file = unpickle(DATAPATH + D_BATCH[i])
        data_train = np.vstack((data_train, file['data']))  # Vertically stack data
        labels_train = np.hstack((labels_train, file['labels']))  # Horizontally stack labels
    # Compose validation data
    indices_val = np.random.choice(range(data_train.shape[0]), size_val, replace=False)  # Select size_val
    # random without repetition
    data_val = data_train[indices_val]  # Copy selected data to validation
    labels_val = labels_train[indices_val]  # Copy selected labels to validation
    data_train = np.delete(data_train, indices_val, axis=0)  # Remove selected data
    labels_train = np.delete(labels_train, indices_val)  # Remove selected labels
    # Reading test data
    file = unpickle(DATAPATH + T_BATCH)
    data_test = file['data']  # Images data for testing
    labels_test = file['labels']  # Images labels for testing
    file = unpickle(DATAPATH + 'batches.meta')
    label_names = file['label_names']  # Images class of each label

    return data_train, labels_train, data_val, labels_val, data_test, labels_test, label_names


def initialize_network(data_train, layer_nodes, layers=2, he=False, sigma=None):
    weights = list()
    bias = list()
    gamma = list()
    beta = list()
    if he:
        num = 2
    else:
        num = 1
    # Weights and bias generation:
    # 1st layer
    if sigma is not None:
        weights.append(np.random.normal(0, sigma, (layer_nodes[0], data_train.shape[0])))  # Dim: m x d
    else:
        weights.append(np.random.normal(0, np.sqrt(num / data_train.shape[0]),
                                        (layer_nodes[0], data_train.shape[0])))  # Dim: m x d

    bias.append(np.zeros((layer_nodes[0], 1)))  # Dim: m x 1
    for i in range(1, layers):  # Remaining layers
        if sigma is not None:
            weights.append(np.random.normal(0, sigma, (layer_nodes[i], weights[-1].shape[0])))  # Dim: l x m
        else:
            weights.append(np.random.normal(0, np.sqrt(num / weights[-1].shape[0]),
                                            (layer_nodes[i], weights[-1].shape[0])))  # Dim: l x m
        bias.append(np.zeros((layer_nodes[i], 1)))  # Dim: l x 1
    # Generate gamma and beta: n x 1 vectors
    for l_nodes in layer_nodes[:-1]:  # Generate for all hidden layers
        gamma.append(np.ones((l_nodes, 1)))  # Dim: m x 1
        beta.append(np.zeros((l_nodes, 1)))  # Dim: m x 1

    return weights, bias, gamma, beta


def add_jitter(data):
    noise = np.random.uniform(np.min(data), np.max(data), (LENGTH, data.shape[1]))  # Create noise for 1/3 of the pixels
    indices = np.random.choice(data.shape[0], noise.shape[0], replace=False)  # Select random positions of the image
    jitter = np.zeros_like(data)
    jitter[indices] = noise
    jitter_img = data + jitter
    jitter_img[jitter_img > np.max(data)] = np.max(data)  # Avoid value overflow
    jitter_img[jitter_img < np.min(data)] = np.min(data)  # Avoid value underflow

    return jitter_img


def batch_normalize(s, mean, var):
    return np.diag(pow(var + np.finfo(float).eps, -1 / 2)) @ (s - mean[:, np.newaxis])


def batch_normalize_back_pass(g, s, mean, var):
    sigma_1 = np.power(var + np.finfo(float).eps, -0.5).T[:, np.newaxis]
    sigma_2 = np.power(var + np.finfo(float).eps, -1.5).T[:, np.newaxis]
    g1 = g * sigma_1
    g2 = g * sigma_2
    d = s - mean[:, np.newaxis]
    c = np.sum(g2 * d, axis=1)[:, np.newaxis]
    g_batch = g1 - (1 / g.shape[1]) * np.sum(g1, axis=1)[:, np.newaxis] - (1 / g.shape[1]) * d * c

    return g_batch


def forward_pass(data_train, weights, bias, gamma=None, beta=None, mean=None, var=None, b_norm=False, dropout=1.0):
    if not b_norm:
        output = list()  # Output of previous layer list (take data as the first output)
        s_list = list()  # s values list
        output.append(np.copy(data_train))  # Append data train as first output
        for i in range(len(weights) - 1):
            s_list.append(compute_s(output[-1], weights[i], bias[i]))
            if dropout != 1.0:
                h = compute_h(s_list[-1])  # Calculate h value using s
                u = np.random.choice([0, 1], size=h.shape, p=[1 - dropout, dropout]) / dropout  # Choose dropped nodes
                output.append(h * u)  # Apply dropout to h
            else:
                output.append(compute_h(s_list[-1]))
        s_list.append(compute_s(output[-1], weights[-1], bias[-1]))
        output.append(softmax(s_list[-1]))  # Vector of probabilities p

        return output, s_list

    else:
        output = list()  # Output of previous layer list (take data as the first output)
        s_list = list()  # s values list
        s_hat_list = list()  # s hat values list
        mean_list = list()  # mean of s values list
        var_list = list()  # variance of s values list
        output.append(np.copy(data_train))  # Append data train as first output
        for i in range(len(weights) - 1):
            s_list.append(compute_s(output[-1], weights[i], bias[i]))
            if mean is None and var is None:
                mean_list.append(np.mean(s_list[-1], axis=1, dtype=np.float64))  # Calculate mean per dimension
                var_list.append(np.var(s_list[-1], axis=1, dtype=np.float64))  # Calculate variance per dimension
            else:
                mean_list.append(mean[i])
                var_list.append(var[i])
            s_hat_list.append(batch_normalize(s_list[-1], mean_list[-1], var_list[-1]))
            s_tilde = gamma[i] * s_hat_list[-1] + beta[i]
            if dropout != 1.0:
                h = compute_h(s_tilde)  # Calculate h value using s_tilde
                u = np.random.choice([0, 1], size=h.shape, p=[1 - dropout, dropout]) / dropout  # Choose dropped nodes
                output.append(h * u)  # Apply dropout to h
            else:
                output.append(compute_h(s_tilde))  # Calculate new input by means of s_tilde
        # Last layer
        s_list.append(compute_s(output[-1], weights[-1], bias[-1]))
        output.append(softmax(s_list[-1]))  # Vector of probabilities p

        return output, s_list, s_hat_list, mean_list, var_list


def cyclical_update(t, n_s, eta_min, eta_max):
    cycle = int(t / (2 * n_s))  # Number of complete cycles elapsed
    if 2 * cycle * n_s <= t <= (2 * cycle + 1) * n_s:
        return eta_min + (t - 2 * cycle * n_s) / n_s * (eta_max - eta_min)
    if (2 * cycle + 1) * n_s <= t <= 2 * (cycle + 1) * n_s:
        return eta_max - (t - (2 * cycle + 1) * n_s) / n_s * (eta_max - eta_min)


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def compute_h(s):
    return np.maximum(0, s)


def l_cross(y, p):
    return -np.log(np.sum(y * p, axis=0))


def compute_s(data, weight, bias):
    return weight @ data + bias  # Dim: k x n


def compute_loss(data, labels, weights, bias, gamma=None, beta=None, mean=None, var=None, b_norm=False):
    p = forward_pass(data, weights, bias, gamma, beta, mean, var, b_norm)[0][-1]  # Value of p
    l_cross_sum = np.sum(l_cross(labels, p))

    return (1 / data.shape[1]) * l_cross_sum


def compute_cost(data, labels, weights, bias, lmb, gamma=None, beta=None, mean=None, var=None, b_norm=False):
    loss = compute_loss(data, labels, weights, bias, gamma, beta, mean, var, b_norm)
    reg = lmb * np.sum([np.sum(np.square(w)) for w in weights])  # Regularization term L2

    return loss + reg


def compute_accuracy(data, labels, weights, bias, gamma=None, beta=None, mean=None, var=None, b_norm=False):
    p = forward_pass(data, weights, bias, gamma, beta, mean, var, b_norm)[0][-1]  # Value of p
    prediction = np.argmax(p, axis=0)
    real = np.argmax(labels, axis=0)

    return np.sum(real == prediction) / len(real)


def compute_grads_analytic(data, labels, weights, lmb, p, s_list, s_hat=None, gamma=None, mean=None, var=None,
                           b_norm=False):
    grad_weights = list()
    grad_bias = list()
    if not b_norm:
        # Last layer --> data[0] is the original input
        g = -(labels - p)  # Dim: k x n
        # Remaining layers
        for i in reversed(range(len(weights))):  # Reverse traversal of the lists
            grad_weights.append((g @ data[i].T) / data[0].shape[1] + 2 * lmb * weights[i])
            grad_bias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
            g = weights[i].T @ g  # Multiply by previous weight
            diag = np.copy(data[i])  # Perform a copy of the output of the previous layer
            diag[diag > 0] = 1  # Transform every element > 0 into 1
            g = g * diag  # Element multiplication by diagonal of the indicator
        grad_weights.reverse(), grad_bias.reverse()  # Reverse lists to return the same order

        return grad_weights, grad_bias

    else:
        grad_gamma = list()
        grad_beta = list()
        # Last layer --> data[0] is the original input
        g = -(labels - p)  # Dim: k x n
        grad_weights.append((g @ data[-2].T) / data[0].shape[1] + 2 * lmb * weights[-1])
        grad_bias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
        g = weights[-1].T @ g  # Multiply by previous weight
        diag = np.copy(data[-2])  # Perform a copy of the output of the previous layer
        diag[diag > 0] = 1  # Transform every element > 0 into 1
        g = g * diag  # Element multiplication by diagonal of the indicator
        # Remaining layers
        for i in reversed(range(len(weights) - 1)):  # Reverse traversal of the lists
            # Gradients for the scale and offset parameters
            grad_gamma.append(np.sum(g * s_hat[i], axis=1)[:, np.newaxis] / data[0].shape[1])
            grad_beta.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
            g = g * gamma[i]
            g = batch_normalize_back_pass(g, s_list[i], mean[i], var[i])
            # Update of the previous layer
            grad_weights.append((g @ data[i].T) / data[0].shape[1] + 2 * lmb * weights[i])
            grad_bias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
            if i > 0:
                g = weights[i].T @ g  # Multiply by previous weight
                diag = np.copy(data[i])  # Perform a copy of the output of the previous layer
                diag[diag > 0] = 1  # Transform every element > 0 into 1
                g = g * diag  # Element multiplication by diagonal of the indicator over data[i]

        # Reverse lists to return the same order
        grad_weights.reverse(), grad_bias.reverse(), grad_gamma.reverse(), grad_beta.reverse()

        return grad_weights, grad_bias, grad_gamma, grad_beta


def update_parameters(item, eta, delta):
    return [item[i] - eta * delta[i] for i in range(len(item))]


def preprocess_images(data, mean, std):
    data = np.float64(data)  # Conversion of data to float64 to perform needed calculations
    if mean is None and std is None:
        mean = np.mean(data, axis=0)  # Mean of the columns
        std = np.std(data, axis=0)  # Std of the columns
    data -= mean
    data /= std

    return np.array(data), mean, std


def one_hot(labels, dim):
    labels_mat = np.zeros((dim, len(labels)))
    for i in range(len(labels)):
        labels_mat[labels[i], i] = 1

    return labels_mat


def preprocess_data(size_val=5000):
    data_train, labels_train, data_val, labels_val, data_test, labels_test, label_names = read_data(size_val)
    data_train, mean_train, std_train = preprocess_images(data_train, mean=None, std=None)  # Preprocess traning data
    data_train = data_train.T  # Transpose data to get the appropriate format --> d x n
    data_val = preprocess_images(data_val, mean_train, std_train)[0].T  # Std. val. using training mean and std
    data_test = preprocess_images(data_test, mean_train, std_train)[0].T  # Std. test using training mean and std
    labels_train = one_hot(labels_train, len(label_names))  # Convert training labels to one-hot matrix
    labels_val = one_hot(labels_val, len(label_names))  # Convert validation labels to one-hot matrix
    labels_test = one_hot(labels_test, len(label_names))  # Convert test labels to one-hot matrix

    return data_train, labels_train, data_val, labels_val, data_test, labels_test, label_names


def get_images(data):
    result = list()
    for i in range(data.shape[0]):
        result.append(read_image(data[i]))

    return result


def read_image(colors):
    red = np.array(colors[0:LENGTH]).reshape(SIZE, SIZE) / 255.0
    green = np.array(colors[LENGTH:2 * LENGTH]).reshape(SIZE, SIZE) / 255.0
    blue = np.array(colors[2 * LENGTH:3 * LENGTH]).reshape(SIZE, SIZE) / 255.0

    return np.dstack((red, green, blue))  # Combine the three color channels


def visualize_images(images, labels, label_names, number=5):
    ig, axes = plt.subplots(number, number)
    indices = np.random.choice(range(len(images)), pow(number, 2))
    labels_aux = np.argmax(labels, axis=0)
    for i in range(number):
        for j in range(number):
            axes[i, j].set_axis_off()
            axes[i, j].text(0.5, -0.5, 'Category: ' + str(label_names[labels_aux[indices[i * number + j]]]),
                            size=6, ha="center", transform=axes[i, j].transAxes)
            axes[i, j].imshow(images[indices[i * number + j]], interpolation='bicubic')
    plt.show()


def plot_results(train, val, mode):
    plt.plot(range(len(train)), train, label="Training " + mode, color="Green")
    plt.plot(range(len(val)), val, label="Validation " + mode, color="Red")
    plt.xlabel("Epoch")
    plt.ylabel(mode.capitalize())
    plt.legend()
    plt.show()


def train_network(data_train, labels_train, data_val, labels_val,
                  weights, bias, gamma, beta, n_batch, eta, n_s, eta_min, eta_max, cycles=2,
                  plotting=False, best_lambda=None, lmb_search=True, alpha=0.9, b_norm=False,
                  dropout=1.0, jitter=False):
    if lmb_search:
        if best_lambda is None:
            l_val = np.random.uniform(-3, -2)  # Random sample from -5 to -1 interval in log10 scale
            lmb = pow(10, l_val)  # Get random lambda
        else:
            dist = abs(best_lambda[0] - best_lambda[1])  # Distance interval
            l_val = np.random.uniform(best_lambda[0] - dist, best_lambda[0] + dist)  # Random sample from
            # the interval defined by the two best previous lambdas in log10 scale
            lmb = pow(10, l_val)  # Define lambda
    else:
        lmb = pow(10, best_lambda)
    iterations = cycles * 2 * n_s  # Number of eta updates
    cycles_per_epoch = data_train.shape[1] / n_batch  # Number of eta update cycles per epoch
    n_epoch = iterations / cycles_per_epoch  # Define number of epochs needed to perform "cycles" updates
    training_loss = list()  # Training data loss per epoch
    validation_loss = list()  # Validation data loss per epoch
    training_cost = list()  # Training data cost per epoch
    validation_cost = list()  # Validation data cost per epoch
    acc_training = list()
    acc_val = list()
    final_mean = list()  # Final avg mean value
    final_var = list()  # Final avg variance value
    for t in tqdm(range(int(n_epoch))):
        mean_av = list()  # Mean average list
        var_av = list()  # Variance average list
        for j in range(int(data_train.shape[1] / n_batch)):
            start = j * n_batch
            end = (j + 1) * n_batch
            if not b_norm:
                data, s_list = forward_pass(data_train[:, start:end] if not jitter else
                                            add_jitter(data_train[:, start:end]),
                                            weights, bias, dropout=dropout)
                delta_w, delta_b = compute_grads_analytic(data, labels_train[:, start:end], weights, lmb,
                                                          data[-1], s_list)
                # Update parameters with the gradients
                weights = update_parameters(weights, eta, delta_w)
                bias = update_parameters(bias, eta, delta_b)

            else:
                data, s_list, s_hat_list, mean_list, var_list = \
                    forward_pass(data_train[:, start:end] if not jitter else
                                 add_jitter(data_train[:, start:end]),
                                 weights, bias, gamma, beta, None, None, b_norm,
                                 dropout=dropout)
                delta_w, delta_b, delta_g, delta_bt = compute_grads_analytic(data, labels_train[:, start:end],
                                                                             weights, lmb, data[-1], s_list, s_hat_list,
                                                                             gamma, mean_list, var_list, b_norm)
                # Update parameters with the gradients
                weights = update_parameters(weights, eta, delta_w)
                bias = update_parameters(bias, eta, delta_b)
                gamma = update_parameters(gamma, eta, delta_g)
                beta = update_parameters(beta, eta, delta_bt)
                # Update weighted avg of mean and variance
                if j == 0:  # First minibatch
                    mean_av = mean_list
                    var_av = var_list
                else:
                    mean_av = [[alpha * x for x in mean_av][y] + [(1 - alpha) * x for x in mean_list][y]
                               for y in range(len(mean_list))]
                    var_av = [[alpha * x for x in var_av][y] + [(1 - alpha) * x for x in var_list][y]
                              for y in range(len(var_list))]
                if t == n_epoch - 1:  # Last iteration of training
                    final_mean = mean_av
                    final_var = var_av
            eta = cyclical_update((t * cycles_per_epoch) + j, n_s, eta_min, eta_max)  # Cyclical uptadte of eta
        if plotting and b_norm:
            training_loss.append(compute_loss(data_train, labels_train, weights, bias, gamma, beta, mean_av, var_av,
                                              b_norm))
            validation_loss.append(compute_loss(data_val, labels_val, weights, bias, gamma, beta, mean_av, var_av,
                                                b_norm))
            training_cost.append(compute_cost(data_train, labels_train, weights, bias, lmb, gamma, beta, mean_av,
                                              var_av, b_norm))
            validation_cost.append(compute_cost(data_val, labels_val, weights, bias, lmb, gamma, beta, mean_av,
                                                var_av, b_norm))
            acc_training.append(compute_accuracy(data_train, labels_train, weights, bias, gamma, beta, mean_av, var_av,
                                                 b_norm))
            acc_val.append(compute_accuracy(data_val, labels_val, weights, bias, gamma, beta, mean_av, var_av, b_norm))
        elif plotting:
            training_loss.append(compute_loss(data_train, labels_train, weights, bias))
            validation_loss.append(compute_loss(data_val, labels_val, weights, bias))
            training_cost.append(compute_cost(data_train, labels_train, weights, bias, lmb))
            validation_cost.append(compute_cost(data_val, labels_val, weights, bias, lmb))
            acc_training.append(compute_accuracy(data_train, labels_train, weights, bias))
            acc_val.append(compute_accuracy(data_val, labels_val, weights, bias))

    # Show results
    if plotting:
        plot_results(acc_training, acc_val, "accuracy")
        plot_results(training_loss, validation_loss, "loss")
        plot_results(training_cost, validation_cost, "cost")

    if b_norm:
        val_acc = compute_accuracy(data_val, labels_val, weights, bias, gamma, beta, final_mean, final_var, b_norm)
        return weights, bias, gamma, beta, final_mean, final_var, val_acc, np.log10(lmb)

    else:
        val_acc = compute_accuracy(data_val, labels_val, weights, bias)
        return weights, bias, val_acc, np.log10(lmb)


def main():
    np.random.seed(40)
    # Read data
    data_train, labels_train, data_val, labels_val, data_test, labels_test, label_names = preprocess_data(size_val=5000)
    weights, bias, gamma, beta = initialize_network(data_train, [200, 50, 10], 3, he=True, sigma=None)
    n_batch = 100  # Define minibatch size
    eta_min = 1e-5  # Minimum value of eta
    eta_max = 1e-1  # Maximum value of eta
    eta = eta_min  # Define learning rate
    n_s = 5 * int(data_train.shape[1] / n_batch)  # Step size in eta value modification

    # Perform training in order to get the best lambda
    lmb_search = 6  # Number of lambda search
    n_lmb = 3  # Best n lambdas to save
    best_acc = np.zeros(lmb_search)  # The accuracies list
    best_lmb = np.zeros(lmb_search)  # The lambdas list
    # First lambda search
    if not os.path.isfile('Data_bonus/data1.npz'):
        for i in range(lmb_search):
            acc, lmb = train_network(data_train, labels_train, data_val, labels_val,
                                     weights, bias, gamma, beta, n_batch, eta, n_s,
                                     eta_min, eta_max, cycles=2,
                                     plotting=False, best_lambda=None, lmb_search=True, b_norm=True,
                                     dropout=0.8)[6:]
            best_acc[i] = acc
            best_lmb[i] = lmb
        indices = find_best_n(best_acc, n_lmb)
        best_acc = best_acc[indices]  # Get the three best accuracies
        best_lmb = best_lmb[indices]  # Get the three best lambdas
        np.savez_compressed('Data_bonus/data1.npz', acc=best_acc, lmb=best_lmb)
    else:
        dict_data = np.load('Data_bonus/data1.npz', allow_pickle=True)
        best_acc = dict_data['acc']
        best_lmb = dict_data['lmb']
    print("Best accuracies in validation: " + str(best_acc))
    print("Best lambdas in validation: " + str(best_lmb))
    # Second lambda search
    if not os.path.isfile('Data_bonus/data2.npz'):
        improved_acc = np.zeros(lmb_search + n_lmb)
        improved_lmb = np.zeros(lmb_search + n_lmb)
        improved_acc[0:n_lmb] = best_acc
        improved_lmb[0:n_lmb] = best_lmb
        for i in range(n_lmb, lmb_search + n_lmb):
            acc, lmb = train_network(data_train, labels_train, data_val, labels_val,
                                     weights, bias, gamma, beta, n_batch, eta, n_s,
                                     eta_min, eta_max, cycles=2,
                                     plotting=False, best_lambda=best_lmb[:2], lmb_search=True, b_norm=True,
                                     dropout=0.8)[6:]
            improved_acc[i] = acc
            improved_lmb[i] = lmb
        indices = find_best_n(improved_acc, n_lmb)
        improved_acc = improved_acc[indices]  # Get the three best accuracies
        improved_lmb = improved_lmb[indices]  # Get the three best lambdas
        np.savez_compressed('Data_bonus/data2.npz', acc=improved_acc, lmb=improved_lmb)
    else:
        dict_data = np.load('Data_bonus/data2.npz', allow_pickle=True)
        improved_acc = dict_data['acc']
        improved_lmb = dict_data['lmb']
    print("Improved accuracies in validation: " + str(improved_acc))
    print("Improved lambdas in validation: " + str(improved_lmb))

    # Final training
    weights, bias, gamma, beta, mean, var = train_network(data_train, labels_train, data_val, labels_val,
                                                          weights, bias, gamma, beta, n_batch, eta, n_s,
                                                          eta_min, eta_max, cycles=3, plotting=True,
                                                          best_lambda=improved_lmb[0], lmb_search=False,
                                                          b_norm=True, dropout=1.0, jitter=True)[:6]
    # Check accuracy over test data
    print("Accuracy on test data: " + str(compute_accuracy(data_test, labels_test, weights, bias,
                                                           gamma, beta, mean, var, b_norm=True) * 100) + "%")


if __name__ == "__main__":
    main()
