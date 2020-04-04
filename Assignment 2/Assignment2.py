import numpy as np
import matplotlib.pyplot as plt
import os.path
from tqdm import tqdm

DATAPATH = 'Datasets/cifar-10-batches-py/'
LENGTH = 1024  # Number of pixels of the image
SIZE = 32  # Pixel dimension of the image
HIDDEN_NODES = 50  # Number of nodes in the hidden layer
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


def initialize_network(data_train, label_names):
    weights = list()
    bias = list()
    weights.append(np.random.normal(0, 1 / np.sqrt(data_train.shape[0]),
                                    (HIDDEN_NODES, data_train.shape[0])))  # Dim: m x d
    weights.append(np.random.normal(0, 1 / np.sqrt(HIDDEN_NODES),
                                    (len(label_names), HIDDEN_NODES)))  # Dim: k x m
    bias.append(np.zeros((HIDDEN_NODES, 1)))  # Dim: m x 1
    bias.append(np.zeros((len(label_names), 1)))  # Dim: k x 1

    return weights, bias


def forward_pass(data_train, weights, bias):
    output = list()  # Output of previous layer list
    s_list = list()  # s values list
    output.append(np.copy(data_train))
    s_list.append(compute_s(data_train, weights[0], bias[0]))
    for i in range(1, len(weights)):
        output.append(compute_h(s_list[-1]))
        s_list.append(compute_s(output[-1], weights[i], bias[i]))

    return output, s_list


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
    s = weight @ data + bias  # Dim: k x n

    return s


def compute_loss(data, labels, weights, bias):
    p = softmax(forward_pass(data, weights, bias)[1][-1])  # Value of s_2 computed in the forward pass
    l_cross_sum = np.sum(l_cross(labels, p))

    return (1 / data.shape[1]) * l_cross_sum


def compute_cost(data, labels, weights, bias, lmb):
    loss = compute_loss(data, labels, weights, bias)
    reg = lmb * np.sum([np.sum(np.square(w)) for w in weights])  # Regularization term L2

    return loss + reg


def compute_accuracy(data, labels, weights, bias):
    p = softmax(forward_pass(data, weights, bias)[1][-1])  # Value of s_2 computed in the forward pass
    prediction = np.argmax(p, axis=0)
    real = np.argmax(labels, axis=0)

    return np.sum(real == prediction) / len(real)


def compute_grads_analytic(data, labels, weights, lmb, p):
    grad_weights = list()
    grad_bias = list()
    # Last layer --> data[0] is the original input
    g = -(labels - p)  # Dim: k x n
    grad_weights.append((g @ data[-1].T) / data[0].shape[1] + 2 * lmb * weights[-1])
    grad_bias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
    # Remaining layers
    for i in reversed(range(len(data) - 1)):  # Reverse traversal of the lists
        g = weights[i + 1].T @ g  # Multiply by previous weight
        diag = np.copy(data[i + 1])  # Perform a copy of the output of the previous layer
        diag[diag > 0] = 1  # Transform every element > 0 into 1
        # diag[diag < 0] = 0  # Transform every element < 0 into 0
        g = g * diag  # Element multiplication by diagonal of the indicator over data[i]
        grad_weights.append((g @ data[i].T) / data[0].shape[1] + 2 * lmb * weights[i])
        grad_bias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
    grad_weights.reverse(), grad_bias.reverse()  # Reverse lists to return the same order

    return grad_weights, grad_bias


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


def visualize_weight(weight, label_names):
    images = list()
    for i in range(weight.shape[0]):
        red = np.array(weight[i][0:LENGTH]).reshape(SIZE, SIZE)
        green = np.array(weight[i][LENGTH:2 * LENGTH]).reshape(SIZE, SIZE)
        blue = np.array(weight[i][2 * LENGTH:3 * LENGTH]).reshape(SIZE, SIZE)
        img = np.dstack((red, green, blue))
        images.append((img - np.min(img)) / (np.max(img) - np.min(img)))
    ig, axes = plt.subplots(2, int(weight.shape[0] / 2))
    for i in range(axes.shape[0]):
        for j in range(axes.shape[1]):
            axes[i, j].set_axis_off()
            axes[i, j].text(0.5, -0.25, 'Category: ' + str(label_names[i * axes.shape[1] + j]),
                            size=8, ha="center", transform=axes[i, j].transAxes)
            axes[i, j].imshow(images[i * axes.shape[1] + j], interpolation='bicubic')
    plt.show()


def plot_results(train, val, mode):
    plt.plot(range(len(train)), train, label="Training " + mode, color="Green")
    plt.plot(range(len(val)), val, label="Validation " + mode, color="Red")
    plt.xlabel("Epoch")
    plt.ylabel(mode.capitalize())
    plt.legend()
    plt.show()


def train_network(data_train, labels_train, data_val, labels_val,
                  weights, bias, n_batch, eta, n_s, eta_min, eta_max, cycles=2,
                  plotting=False, best_lambda=None, lmb_search=True):
    if lmb_search:
        if best_lambda is None:
            l_val = np.random.uniform(-5, -1)  # Random sample from -5 to -1 interval in log10 scale
            lmb = pow(10, l_val)  # Get random lambda
        else:
            l_val = np.random.uniform(best_lambda[0], best_lambda[1])  # Random sample from
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
    eta_val = list()  # Value of eta per cycle
    for t in tqdm(range(int(n_epoch))):
        for j in range(int(data_train.shape[1] / n_batch)):
            eta_val.append(eta)
            start = j * n_batch
            end = (j + 1) * n_batch
            data, s_list = forward_pass(data_train[:, start:end], weights, bias)
            delta_w, delta_b = compute_grads_analytic(data, labels_train[:, start:end],
                                                      weights, lmb, softmax(s_list[-1]))
            weights = [weights[i] - eta * delta_w[i] for i in range(len(weights))]
            bias = [bias[i] - eta * delta_b[i] for i in range(len(bias))]
            eta = cyclical_update((t * cycles_per_epoch) + j, n_s, eta_min, eta_max)  # Cyclical uptadte of eta
        if plotting:
            training_loss.append(compute_loss(data_train, labels_train, weights, bias))
            validation_loss.append(compute_loss(data_val, labels_val, weights, bias))
            training_cost.append(compute_cost(data_train, labels_train, weights, bias, lmb))
            validation_cost.append(compute_cost(data_val, labels_val, weights, bias, lmb))
            acc_training.append(compute_accuracy(data_train, labels_train, weights, bias))
            acc_val.append(compute_accuracy(data_val, labels_val, weights, bias))

    # Show results
    if plotting:
        plt.plot(range(len(eta_val)), eta_val)
        plt.xlabel("Update step")
        plt.ylabel(r"$\eta_{value}$")
        plt.show()
        plot_results(acc_training, acc_val, "accuracy")
        plot_results(training_loss, validation_loss, "loss")
        plot_results(training_cost, validation_cost, "cost")

    return weights, bias, compute_accuracy(data_val, labels_val, weights, bias), np.log10(lmb)


def main():
    np.random.seed(8)
    # Read data
    data_train, labels_train, data_val, labels_val, data_test, labels_test, label_names = preprocess_data(size_val=5000)
    # Initialize model parameters
    weights, bias = initialize_network(data_train, label_names)
    n_batch = 100  # Define minibatch size
    eta_min = 1e-5  # Minimum value of eta
    eta_max = 1e-1  # Maximum value of eta
    eta = eta_min  # Define learning rate
    n_s = 2 * int(data_train.shape[1] / n_batch)  # Step size in eta value modification

    # Perform training in order to get the best lambda
    lmb_search = 8  # Number of lambda search
    best_acc = np.zeros(lmb_search)  # The accuracies list
    best_lmb = np.zeros(lmb_search)  # The lambdas list
    # First lambda search
    if not os.path.isfile('Data/data1.npz'):
        for i in range(lmb_search):
            acc, lmb = train_network(data_train, labels_train, data_val, labels_val,
                                     weights, bias, n_batch, eta, n_s, eta_min, eta_max, cycles=2,
                                     plotting=False, best_lambda=None, lmb_search=True)[2:]
            best_acc[i] = acc
            best_lmb[i] = lmb
        indices = find_best_n(best_acc, 3)
        best_acc = best_acc[indices]  # Get the three best accuracies
        best_lmb = best_lmb[indices]  # Get the three best lambdas
        np.savez_compressed('Data/data1.npz', acc=best_acc, lmb=best_lmb)
    else:
        dict_data = np.load('Data/data1.npz', allow_pickle=True)
        best_acc = dict_data['acc']
        best_lmb = dict_data['lmb']
    print("Best accuracies in validation: " + str(best_acc))
    print("Best lambdas in validation: " + str(best_lmb))
    # Second lambda search
    if not os.path.isfile('Data/data2.npz'):
        improved_acc = np.zeros(lmb_search + 1)
        improved_lmb = np.zeros(lmb_search + 1)
        improved_acc[0] = best_acc[0]
        improved_lmb[0] = best_lmb[0]
        for i in range(1, lmb_search):
            acc, lmb = train_network(data_train, labels_train, data_val, labels_val,
                                     weights, bias, n_batch, eta, n_s, eta_min, eta_max, cycles=4,
                                     plotting=False, best_lambda=best_lmb[:2], lmb_search=True)[2:]
            improved_acc[i] = acc
            improved_lmb[i] = lmb
        indices = find_best_n(improved_acc, 3)
        improved_acc = improved_acc[indices]  # Get the three best accuracies
        improved_lmb = improved_lmb[indices]  # Get the three best lambdas
        np.savez_compressed('Data/data2.npz', acc=improved_acc, lmb=improved_lmb)
    else:
        dict_data = np.load('Data/data2.npz', allow_pickle=True)
        improved_acc = dict_data['acc']
        improved_lmb = dict_data['lmb']
    print("Improved accuracies in validation: " + str(improved_acc))
    print("Improved lambdas in validation: " + str(improved_lmb))
    # Training with the best found parameters
    indices_val = np.random.choice(range(data_val.shape[1]), 4000, replace=False)  # Select random samples in validation
    data_train = np.hstack((data_train, data_val[:, indices_val]))  # Add previous samples to the training set
    labels_train = np.hstack((labels_train, labels_val[:, indices_val]))  # Add correspondent labels
    data_val = np.delete(data_val, indices_val, axis=1)  # Delete selected samples from validation
    labels_val = np.delete(labels_val, indices_val, axis=1)  # Delete correspondent labels
    n_s = 4 * int(data_train.shape[1] / n_batch)  # Step size in eta value modification
    weights, bias = train_network(data_train, labels_train, data_val, labels_val,
                                  weights, bias, n_batch, eta, n_s, eta_min, eta_max, cycles=3,
                                  plotting=True, best_lambda=improved_lmb[0], lmb_search=False)[0:2]
    # Check accuracy over test data
    print("Accuracy on test data: " + str(compute_accuracy(data_test, labels_test, weights, bias) * 100) + "%")


if __name__ == "__main__":
    main()