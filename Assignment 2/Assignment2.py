import numpy as np
import matplotlib.pyplot as plt
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
    output.append(data_train)
    s_list.append(compute_s(data_train, weights[0], bias[0]))
    for i in range(1, len(weights)):
        output.append(compute_h(s_list[-1]))
        s_list.append(compute_s(output[-1], weights[i], bias[i]))

    return output, s_list



def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def compute_h(s):
    return np.maximum(0, s)


def l_cross(y, p):
    return -np.log(np.sum(y * p, axis=0))


def compute_s(data, weight, bias):
    s = weight @ data + bias  # Dim: k x n

    return s


def compute_loss(data, labels, p):
    l_cross_sum = l_cross(labels, p)
    l_cross_sum = np.sum(l_cross_sum)

    return (1 / data.shape[1]) * l_cross_sum


def compute_cost(data, labels, weights, lmb, p):
    loss = compute_loss(data, labels, p)
    reg = lmb * np.sum([np.sum(np.square(w)) for w in weights])   # Regularization term L2

    return loss + reg


def compute_accuracy(labels, p):
    prediction = np.argmax(p, axis=0)
    real = np.argmax(labels, axis=0)

    return np.sum(real == prediction) / len(real)


def compute_grads_analytic(data, s, labels, weights, lmb, p):
    grad_weights = list()
    grad_bias = list()

    # Last layer --> data[0] is the original input
    g = -(labels - p)  # Dim: k x n
    grad_weights.append((g @ data[-1].T) / data[0].shape[1] + 2 * lmb * weights[-1])
    grad_bias.append(np.sum(g, axis=1)[:, np.newaxis] / data[0].shape[1])
    # Remaining layers
    for i in reversed(range(len(data) - 1)):  # Reverse traversal of the lists
        g = weights[i + 1].T @ g  # Multiply by previous weight
        diag = np.copy(s[i])  # Perform a copy of the output of the previous layer
        diag[diag > 0] = 1  # Transform every element > 0 into 1
        diag[diag < 0] = 0  # Transform every element < 0 into 0
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


def main():
    np.random.seed(42)
    # Reading training and validation data
    file = unpickle(DATAPATH + D_BATCH[0])
    data_train = file['data']  # Images data for training
    labels_train = file['labels']  # Images labels for training
    file = unpickle(DATAPATH + D_BATCH[1])
    data_val = file['data']  # Images data for validation
    labels_val = file['labels']  # Images labels for validation
    # Reading test data
    file = unpickle(DATAPATH + T_BATCH)
    data_test = file['data']  # Images data for testing
    labels_test = file['labels']  # Images labels for testing
    file = unpickle(DATAPATH + 'batches.meta')
    label_names = file['label_names']  # Images class of each label

    # Data preprocessing
    data_train, mean_train, std_train = preprocess_images(data_train, mean=None, std=None)  # Preprocess traning data
    data_train = data_train.T  # Transpose data to get the appropriate format --> d x n
    data_val = preprocess_images(data_val, mean_train, std_train)[0].T  # Std. val. using training mean and std
    data_test = preprocess_images(data_test, mean_train, std_train)[0].T  # Std. test using training mean and std
    labels_train = one_hot(labels_train, len(label_names))  # Convert training labels to one-hot matrix
    labels_val = one_hot(labels_val, len(label_names))  # Convert validation labels to one-hot matrix
    labels_test = one_hot(labels_test, len(label_names))  # Convert test labels to one-hot matrix

    # Initialize model parameters
    weights, bias = initialize_network(data_train, label_names)

    n_batch = 100  # Define minibatch size
    n_epoch = 40  # Define number of epochs
    lmb = 1  # Define lambda
    eta = 0.001  # Define learning rate
    training_loss = list()  # Training data loss per epoch
    validation_loss = list()  # Validation data loss per epoch
    training_cost = list()  # Training data cost per epoch
    validation_cost = list()  # Validation data cost per epoch

    # Perform training
    print("Training model...")
    for _ in tqdm(range(n_epoch)):
        data, s_list = forward_pass(data_train, weights, bias)
        for j in range(int(data_train.shape[1] / n_batch)):
            start = j * n_batch
            end = (j + 1) * n_batch
            delta_w, delta_b = compute_grads_analytic([i[:, start:end] for i in data],
                                                      [i[:, start:end] for i in s_list],
                                                      labels_train[:, start:end],
                                                      weights, lmb, softmax(s_list[-1][:, start:end]))
            weights = [weights[i] - eta * delta_w[i] for i in range(len(weights))]
            bias = [bias[i] - eta * delta_b[i] for i in range(len(bias))]
        training_loss.append(compute_loss(data_train, labels_train, softmax(s_list[-1])))
        validation_loss.append(compute_loss(data_val, labels_val, softmax(s_list[-1])))
        training_cost.append(compute_cost(data_train, labels_train, weights, lmb, softmax(s_list[-1])))
        validation_cost.append(compute_cost(data_val, labels_val, weights, lmb, softmax(s_list[-1])))

    # visualize_weight(weights[1], label_names)

    # Show results
    s_2 = forward_pass(data_train, weights, bias)[1][-1]
    plot_results(training_loss, validation_loss, "loss")
    plot_results(training_cost, validation_cost, "cost")
    print("Accuracy on test data: " + str(compute_accuracy(labels_test, softmax(s_2)) * 100) + "%")


if __name__ == "__main__":
    main()
