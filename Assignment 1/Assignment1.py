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


def evaluate_classifier(data, weight, bias):
    x = data.T  # Transpose data to get the appropriate format --> d x n
    s = weight @ x + bias  # Dim: k x n

    return softmax(s)


def compute_cost(data, labels, weight, bias, lmb):
    x = data.T  # Transpose data to get the appropriate format --> d x n
    y = np.array(labels).reshape(1, len(labels))


def preprocess_images(data):
    data = np.float64(data)  # Conversion of data to float64 to perform needed calculations
    mean = np.mean(data, axis=0)  # Mean of the columns
    std = np.std(data, axis=0)  # Std of the columns
    data -= mean
    data /= std

    return np.array(data)


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
    data_train = preprocess_images(data_train)
    
    # Compute probabilities
    prob = evaluate_classifier(data_train, weight, bias)
    compute_cost(data_train, labels_train, weight, bias, 1)


if __name__ == "__main__":
    main()
