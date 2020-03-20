import numpy as np
import matplotlib.pyplot as plt

DATAPATH = 'Datasets/cifar-10-batches-py/'
LENGTH = 1024  # Number of pixels of the image
SIZE = 32  # Pixel dimension of the image
D_BATCH = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
T_BATCH = 'test_batch'
ETA = 0.001


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='latin1')
    return dictionary


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def l_cross(y, p):
    return -np.log(np.sum(y * p, axis=0))


def evaluate_classifier(data, weight, bias):
    s = weight @ data + bias  # Dim: k x n

    return softmax(s)


def compute_cost(data, labels, weight, bias, lmb):
    p = evaluate_classifier(data, weight, bias)  # Dim: k x n
    l_cross_sum = l_cross(labels, p)
    l_cross_sum = np.sum(l_cross_sum)
    reg = lmb * np.sum(np.sum(np.square(weight)))  # Regularization term L2

    return (1 / data.shape[1]) * l_cross_sum + reg


def compute_accuracy(data, labels, weight, bias):
    p = evaluate_classifier(data, weight, bias)  # Dim: k x n
    prediction = np.argmax(p, axis=0)
    real = np.argmax(labels, axis=0)

    return np.sum(real == prediction) / len(real)


def compute_grads_analytic(data, labels, weight, lmb, p):
    grad_bias = -(labels - p)  # Dim: k x n
    grad_weight = grad_bias @ data.T + 2 * lmb * weight
    grad_bias = np.sum(grad_bias, axis=1)[:, np.newaxis]

    return grad_weight / data.shape[1], grad_bias / data.shape[1]


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
    np.random.seed(42)
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
    data_train = data_train.T  # Transpose data to get the appropriate format --> d x n
    data_val = preprocess_images(data_val, mean_train, std_train)[0].T  # Std. val. using training mean and std
    data_test = preprocess_images(data_test, mean_train, std_train)[0].T  # Std. test using training mean and std

    # Convert labels to one-hot matrix
    labels_train = one_hot(labels_train, len(label_names))
    labels_val = one_hot(labels_val, len(label_names))
    labels_test = one_hot(labels_test, len(label_names))

    # compute_cost(data_train, labels_train, weight, bias, 0)

    for i in range(10):
        delta_w, delta_b = compute_grads_analytic(data_train[:, 0:5], labels_train[:, 0:5], weight, 0,
                                                  evaluate_classifier(data_train[:, 0:5], weight, bias))
        weight -= ETA * delta_w
        bias -= ETA * delta_b


if __name__ == "__main__":
    main()
