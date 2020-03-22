import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    return -np.log(np.sum(y * p, axis=0))


def evaluate_classifier(data, weight, bias):
    s = weight @ data + bias  # Dim: k x n

    return softmax(s)


def compute_loss(data, labels, weight, bias):
    p = evaluate_classifier(data, weight, bias)  # Dim: k x n
    l_cross_sum = l_cross(labels, p)
    l_cross_sum = np.sum(l_cross_sum)

    return (1 / data.shape[1]) * l_cross_sum


def compute_cost(data, labels, weight, bias, lmb):
    loss = compute_loss(data, labels, weight, bias)
    reg = lmb * np.sum(np.sum(np.square(weight)))  # Regularization term L2

    return loss + reg


def compute_accuracy(data, labels, weight, bias):
    p = evaluate_classifier(data, weight, bias)  # Dim: k x n
    prediction = np.argmax(p, axis=0)
    real = np.argmax(labels, axis=0)

    return np.sum(real == prediction) / len(real)


def compute_grads_analytic(data, labels, weight, lmb, p):
    grad_bias = -(labels - p)  # Dim: k x n
    grad_weight = (grad_bias @ data.T) / data.shape[1] + 2 * lmb * weight
    grad_bias = np.sum(grad_bias, axis=1)[:, np.newaxis]

    return grad_weight, grad_bias / data.shape[1]


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
    # Data reading
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

    # Obtain images
    # images_train = get_images(data_train)
    # images_val = get_images(data_val)
    # images_test = get_images(data_test)

    # Data preprocessing
    data_train, mean_train, std_train = preprocess_images(data_train, mean=None, std=None)  # Preprocess traning data
    data_train = data_train.T  # Transpose data to get the appropriate format --> d x n
    data_val = preprocess_images(data_val, mean_train, std_train)[0].T  # Std. val. using training mean and std
    data_test = preprocess_images(data_test, mean_train, std_train)[0].T  # Std. test using training mean and std
    labels_train = one_hot(labels_train, len(label_names))  # Convert training labels to one-hot matrix
    labels_val = one_hot(labels_val, len(label_names))  # Convert validation labels to one-hot matrix
    labels_test = one_hot(labels_test, len(label_names))  # Convert test labels to one-hot matrix

    # Initialize model parameters
    weight = np.random.normal(0, 0.01, (len(label_names), data_train.shape[0]))  # Dim: k x d
    bias = np.random.normal(0, 0.01, (len(label_names), 1))  # Dim: k x 1
    n_batch = 100  # Define minibatch size
    n_epoch = 40  # Define number of epochs
    lmb = 0  # Define lambda
    eta = 0.1  # Define learning rate
    training_loss = list()  # Training data loss per epoch
    validation_loss = list()  # Validation data loss per epoch
    training_cost = list()  # Training data cost per epoch
    validation_cost = list()  # Validation data cost per epoch
    
    # Perform training
    print("Training model...")
    for _ in tqdm(range(n_epoch)):
        for j in range(int(data_train.shape[1] / n_batch)):
            start = j * n_batch
            end = (j + 1) * n_batch
            delta_w, delta_b = compute_grads_analytic(data_train[:, start:end], labels_train[:, start:end], weight, lmb,
                                                      evaluate_classifier(data_train[:, start:end], weight, bias))
            weight -= eta * delta_w
            bias -= eta * delta_b
        training_loss.append(compute_loss(data_train, labels_train, weight, bias))
        validation_loss.append(compute_loss(data_val, labels_val, weight, bias))
        training_cost.append(compute_cost(data_train, labels_train, weight, bias, lmb))
        validation_cost.append(compute_cost(data_val, labels_val, weight, bias, lmb))

    visualize_weight(weight, label_names)

    # Show results
    plot_results(training_loss, validation_loss, "loss")
    plot_results(training_cost, validation_cost, "cost")
    print("Accuracy on test data: " + str(compute_accuracy(data_test, labels_test, weight, bias) * 100) + "%")


if __name__ == "__main__":
    main()
