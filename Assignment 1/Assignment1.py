import numpy as np
import matplotlib.pyplot as plt

DATAPATH = 'Datasets/cifar-10-batches-py/'
LENGTH = 1024  # Number of pixels of the image
SIZE = 32  # Pixel dimension of the image
D_BATCH = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
T_BATCH = 'test_batch'


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='latin1')
    return dictionary


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def l_cross(y, p):
    return -np.log(p[y])


def evaluate_classifier(data, weight, bias):
    s = weight @ data + bias  # Dim: k x n

    return softmax(s)


def compute_cost(data, labels, weight, bias, lmb):
    p = evaluate_classifier(data, weight, bias)  # Dim: k x n
    l_cross_sum = 0
    for i in range(data.shape[1]):
        l_cross_sum += l_cross(labels[i], p[:, i])
    reg = lmb * np.sum(np.sum(np.square(weight)))  # Regularization term L2

    return (1 / data.shape[1]) * l_cross_sum + reg


def compute_accuracy(data, labels, weight, bias):
    p = evaluate_classifier(data, weight, bias)  # Dim: k x n
    prediction = np.argmax(p, axis=0)
    real = np.array(labels).reshape(prediction.shape)

    return np.sum(real == prediction) / len(labels)


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


def preprocess_images(data, mean, std):
    data = np.float64(data)  # Conversion of data to float64 to perform needed calculations
    if mean is None and std is None:
        mean = np.mean(data, axis=0)  # Mean of the columns
        std = np.std(data, axis=0)  # Std of the columns
    data -= mean
    data /= std

    return np.array(data), mean, std


def get_images(data):
    result = list()
    for i in range(data.shape[0]):
        result.append(read_image(data[i]))

    return result


def read_image(colors):
    red = np.array(colors[0:LENGTH]).reshape(SIZE, SIZE) / 255.0
    green = np.array(colors[LENGTH:2*LENGTH]).reshape(SIZE, SIZE) / 255.0
    blue = np.array(colors[2*LENGTH:3*LENGTH]).reshape(SIZE, SIZE) / 255.0

    return np.dstack((red, green, blue))  # Combine the three color channels


def visualize_images(images, labels, label_names, number=5):
    ig, axes = plt.subplots(number, number)
    indices = np.random.choice(range(len(images)), pow(number, 2))
    for i in range(number):
        for j in range(number):
            axes[i, j].set_axis_off()
            axes[i, j].text(0.5, -0.5, 'Category: ' + str(label_names[labels[indices[i * number + j]]]),
                            size=6, ha="center", transform=axes[i, j].transAxes)
            axes[i, j].imshow(images[indices[i * number + j]], interpolation='bicubic')
    plt.show()


def main():
    file = unpickle(DATAPATH + D_BATCH[0])
    data_train = file['data']  # Images data for training
    labels_train = file['labels']  # Images labels for training
    file = unpickle(DATAPATH + D_BATCH[1])
    data_val = file['data']  # Images data for validation
    labels_val = file['labels']  # Images labels for validation
    file = unpickle(DATAPATH + D_BATCH[2])
    data_test = file['data']  # Images data for testing
    labels_test = file['labels']  # Images labels for testing
    file = unpickle(DATAPATH + 'batches.meta')
    label_names = file['label_names']  # Images class of each label

    # Initialize model parameters
    weight = np.random.normal(0, 0.01, (len(label_names), data_train.shape[1]))  # Dim: k x d
    bias = np.random.normal(0, 0.01, (len(label_names), 1))  # Dim: k x 1

    # Preprocess traning data
    data_train, mean_train, std_train = preprocess_images(data_train, mean=None, std=None)
    data_train = data_train.T  # Transpose data to get the appropriate format --> d x ns
    data_val = preprocess_images(data_val, mean_train, std_train)[0].T  # Std. val. using training mean and std
    data_test = preprocess_images(data_test, mean_train, std_train)[0].T  # Std. test using training mean and std

    compute_grads_num_slow(data_train, labels_train, weight, bias, 1, 1)


if __name__ == "__main__":
    main()
