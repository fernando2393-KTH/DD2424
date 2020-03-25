import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

DATAPATH = 'Datasets/cifar-10-batches-py/'
LENGTH = 1024  # Number of pixels of the image
SIZE = 32  # Pixel dimension of the image
D_BATCH = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
T_BATCH = 'test_batch'
FULL_TRAIN = False
SIZE_VAL = 1000  # Number of elements in validation set when FULL_TRAIN = True
DECAY = False
SHUFFLE = False
SVM_LOSS = False


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


def compute_loss_svm(data, labels, weight, bias):
    weight_aux = np.hstack((weight, bias))  # Stack bias together with weight
    data_aux = np.vstack((data, np.ones(data.shape[1])))  # Add row of ones for bias
    score = weight_aux @ data_aux  # Compute the score
    labels_aux = np.argmax(labels, axis=0)  # Get the correct class from the one hot matrix
    target_scores = score[labels_aux, np.arange(score.shape[1])]  # Get the score of the correct class of each sample
    margins = np.maximum(0, score - target_scores + 1)  # Compute the SVM margins
    margins[labels_aux, np.arange(data.shape[1])] = 0  # Make zero the own margins
    loss = np.mean(np.sum(margins, axis=0))  # Compute loss as mean of the sum per class

    return loss, margins


def compute_cost(data, labels, weight, bias, lmb):
    loss = compute_loss(data, labels, weight, bias)
    reg = lmb * np.sum(np.square(weight))  # Regularization term L2

    return loss + reg


def compute_cost_svm(data, labels, weight, bias, lmb):
    loss = compute_loss_svm(data, labels, weight, bias)[0]
    reg = lmb * np.sum(np.square(weight))  # Apply same L2 regularization used in compute_cost

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


def compute_grads_analytic_svm(data, labels, weight, bias, lmb):
    margins = compute_loss_svm(data, labels, weight, bias)[1]  # Dim: k x d
    weight_aux = np.hstack((weight, bias))  # Stack bias together with weight
    data_aux = np.vstack((data, np.ones(data.shape[1])))  # Row of ones for the bias term
    labels_aux = np.argmax(labels, axis=0)  # Get the correct class from the one hot matrix
    margins[margins > 0] = 1
    col_sum = np.sum(margins, axis=0)  # Dim: 1 x d
    margins[labels_aux, np.arange(data_aux.shape[1])] = -col_sum
    grad_weight = margins @ data_aux.T

    return grad_weight / data.shape[1] + 2 * lmb * weight_aux


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
    # np.random.seed(42)
    if FULL_TRAIN:
        file = unpickle(DATAPATH + D_BATCH[0])
        data_train = file['data']  # Images data for training
        labels_train = file['labels']  # Images labels for training
        for i in range(1, len(D_BATCH)):  # Compose the training set
            file = unpickle(DATAPATH + D_BATCH[i])
            data_train = np.vstack((data_train, file['data']))  # Vertically stack data
            labels_train = np.hstack((labels_train, file['labels']))  # Horizontally stack labels
        # Compose validation data
        indices_val = np.random.choice(range(data_train.shape[0]), SIZE_VAL, replace=False)  # Select size_val
        # random without repetition
        data_val = data_train[indices_val]  # Copy selected data to validation
        labels_val = labels_train[indices_val]  # Copy selected labels to validation
        data_train = np.delete(data_train, indices_val, axis=0)  # Remove selected data
        labels_train = np.delete(labels_train, indices_val)  # Remove selected labels
    else:
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
    lmb = 1  # Define lambda
    eta = 0.001  # Define learning rate
    training_loss = list()  # Training data loss per epoch
    validation_loss = list()  # Validation data loss per epoch
    training_cost = list()  # Training data cost per epoch
    validation_cost = list()  # Validation data cost per epoch
    
    # Perform training
    print("Training model...")
    for _ in tqdm(range(n_epoch)):
        if SHUFFLE:  # Shuffle training data and labels per epoch
            columns = np.arange(data_train.shape[1])  # Get an array with the indeces of the columns to shuffle
            np.random.shuffle(columns)  # Shuffle indeces
            data_train = data_train[:, columns]  # Shuffle data columns according to the new ordered indeces
            labels_train = labels_train[:, columns]  # Shuffle labels columns according to the new ordered indeces
        for j in range(int(data_train.shape[1] / n_batch)):
            start = j * n_batch
            end = (j + 1) * n_batch
            if SVM_LOSS:
                delta_w = compute_grads_analytic_svm(data_train[:, start:end], labels_train[:, start:end], weight,
                                                     bias, lmb)
                delta_b = (delta_w[:, -1])[:, np.newaxis]
                delta_w = np.delete(delta_w, -1, axis=1)
            else:
                delta_w, delta_b = compute_grads_analytic(data_train[:, start:end], labels_train[:, start:end],
                                                          weight, lmb, evaluate_classifier(data_train[:, start:end],
                                                                                           weight, bias))
            weight -= eta * delta_w
            bias -= eta * delta_b
        if DECAY:
            eta *= 0.9
        if SVM_LOSS:
            training_loss.append(compute_loss_svm(data_train, labels_train, weight, bias)[0])
            validation_loss.append(compute_loss_svm(data_val, labels_val, weight, bias)[0])
            training_cost.append(compute_cost_svm(data_train, labels_train, weight, bias, lmb))
            validation_cost.append(compute_cost_svm(data_val, labels_val, weight, bias, lmb))
        else:
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
