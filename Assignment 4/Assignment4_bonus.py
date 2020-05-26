import numpy as np
import json
import matplotlib.pyplot as plt
import copy
import pandas as pd
from tqdm import tqdm
from os import listdir

PATH = "../Datasets/"
SECTIONS = 5


class RNN:
    def __init__(self, k=1, m=100, seq_length=25, eta=0.1, sig=0.01):
        self.m = m  # Dimensionality of hidden state
        self.eta = eta  # Learning rate
        self.seq_length = seq_length  # Length of training sequences
        self.b = np.zeros((m, 1))  # Bias vector b
        self.c = np.zeros((k, 1))  # Bias vector c
        self.u = np.random.rand(m, k) * sig  # Weight matrix u
        self.w = np.random.rand(m, m) * sig  # Weight matrix w
        self.v = np.random.rand(k, m) * sig  # Weight matrix v


def read_tweets():
    files = sorted(listdir("Tweets/"))
    # files = ["Donald-Tweets!.csv"]
    unique_char = list()
    final_data = list()
    for file in files:
        if file.endswith('.csv'):
            with open("Tweets/" + file, encoding="utf-8") as f:
                # data = json.load(f)
                tweets = list(pd.read_csv(f)['Tweet_Text'].values)
            tweets = [item + '\r' for item in tweets]  # Read all tweets and add an
            # end-of-tweet character at the end of each one
            unique_char.append({char for tweet in tweets for char in tweet})
            final_data = final_data + tweets
    unique_char = set.union(*unique_char)  # Intersection of all unique characters per tweets year

    return final_data, unique_char


def softmax(s):
    return np.exp(s) / np.sum(np.exp(s), axis=0)


def synthesize(rnn, h_0, x_0, n):
    x = np.copy(x_0)
    h = np.copy(h_0)[:, np.newaxis]
    samples = np.zeros((x_0.shape[0], n))
    for t in range(n):
        a = rnn.w @ h + rnn.u @ x + rnn.b
        h = np.tanh(a)
        o = rnn.v @ h + rnn.c
        p = softmax(o)
        choice = np.random.choice(range(x.shape[0]), 1, p=p.flatten())  # Select random character
        # according to probabilities
        x = np.zeros(x.shape)
        x[choice] = 1
        samples[:, t] = x.flatten()

    return samples


def forward(rnn, h_0, x):
    h = np.zeros((h_0.shape[0], x.shape[1]))
    a = np.zeros((h_0.shape[0], x.shape[1]))
    prob = np.zeros(x.shape)
    for t in range(x.shape[1]):
        if t == 0:
            a[:, t] = (rnn.w @ h_0[:, np.newaxis] + rnn.u @ x[:, t][:, np.newaxis] + rnn.b).flatten()
        else:
            a[:, t] = (rnn.w @ h[:, t - 1][:, np.newaxis] + rnn.u @ x[:, t][:, np.newaxis] + rnn.b).flatten()
        h[:, t] = np.tanh(a[:, t])
        o = rnn.v @ h[:, t][:, np.newaxis] + rnn.c
        p = softmax(o)
        prob[:, t] = p.flatten()

    return prob, h, a


def backprop(rnn, y, p, h, h_prev, a, x):
    grad_h = list()
    grad_a = list()
    # Computation of the last gradient of o
    grad_o = -(y - p).T
    # Computation of the last gradients of h and a
    grad_h.append(grad_o[-1][np.newaxis, :] @ rnn.v)
    grad_a.append((grad_h[-1] @ np.diag(1 - np.power(np.tanh(a[:, -1]), 2))))
    # Computation of the remaining gradients of o, h, and a
    for t in reversed(range(y.shape[1] - 1)):
        grad_h.append(grad_o[t][np.newaxis, :] @ rnn.v + grad_a[-1] @ rnn.w)
        grad_a.append(grad_h[-1] @ np.diag(1 - np.power(np.tanh(a[:, t]), 2)))

    grad_a.reverse()  # Reverse a gradient so it goes forwards
    grad_a = np.vstack(grad_a)  # Stack gradients of a as a matrix
    rnn_grads = RNN()  # Define rnn object to store the gradients
    rnn_grads.v = grad_o.T @ h.T
    h_aux = np.zeros(h.shape)  # Auxiliar h matrix that includes h_prev
    h_aux[:, 0] = h_prev
    h_aux[:, 1:] = h[:, 0:-1]
    rnn_grads.w = grad_a.T @ h_aux.T
    rnn_grads.u = grad_a.T @ x.T
    rnn_grads.b = np.sum(grad_a, axis=0)[:, np.newaxis]
    rnn_grads.c = np.sum(grad_o, axis=0)[:, np.newaxis]

    return rnn_grads


def compute_loss(y, p):
    return -np.sum(np.log(np.sum(y * p, axis=0)))


def one_hot(vec, conversor):
    mat = np.zeros((len(conversor), len(vec)))
    for i in range(len(vec)):
        mat[conversor[vec[i]], i] = 1

    return mat


def adagrad(m_old, g, param_old, eta):
    m = m_old + np.power(g, 2)
    param = param_old - (eta / np.sqrt(m + np.finfo('float').eps)) * g

    return param, m


def main():
    np.random.seed(40)
    char_to_ind = {}
    ind_to_char = {}
    # book_chars, book_unique_chars = read_data()
    tweets, book_unique_chars = read_tweets()

    for idx, x in enumerate(book_unique_chars):  # Create the enconding conversors
        char_to_ind[x] = idx
        ind_to_char[idx] = x

    d = len(book_unique_chars)  # Dimensionality --> number of different characters
    rnn = RNN(k=d)  # Initialize recurrent neural network object
    h_prev = np.zeros(rnn.m)  # First hidden state
    loss_list = list()
    smooth_loss = 0
    e = 0
    m_list = [0, 0, 0, 0, 0]
    best_rnn = RNN()
    best_loss = float('inf')
    for epoch in range(10):
        np.random.shuffle(tweets)
        for tweet_idx, tweet in enumerate(tqdm(tweets)):
            seq_length = int(len(tweet) / SECTIONS)
            for it in range(SECTIONS):  # Epoch iteration
                # Choose the sample characters
                if it < SECTIONS - 1:
                    x = one_hot(tweet[e:e + seq_length], char_to_ind)  # Input vector
                    y = one_hot(tweet[e + 1:e + 1 + seq_length], char_to_ind)  # Target output vector
                else:
                    x = one_hot(tweet[e:-1], char_to_ind)  # Input vector
                    y = one_hot(tweet[e + 1:], char_to_ind)  # Target output vector
                p, h, a = forward(rnn, h_prev, x)
                rnn_grads = backprop(rnn, y, p, h, h_prev, a, x)
                # Check exploding gradients
                for idx, att in enumerate(['b', 'c', 'u', 'w', 'v']):
                    grad = getattr(rnn_grads, att)
                    grad = np.clip(grad, -5, 5)
                    att_new, m_val = adagrad(m_list[idx], grad, getattr(rnn, att), rnn.eta)
                    setattr(rnn, att, att_new)
                    m_list[idx] = m_val
                if tweet_idx == 0 and epoch == 0:  # and it == 0:  # Calculate first loss for training
                    smooth_loss = compute_loss(y, p)
                    loss_list.append(smooth_loss)
                    best_rnn = copy.deepcopy(rnn)  # Select first model as the best one
                    best_loss = smooth_loss  # Update best loss with the new one
                else:
                    smooth_loss = 0.999 * smooth_loss + 0.001 * compute_loss(y, p)
                    if smooth_loss < best_loss:  # Check model loss
                        best_rnn = copy.deepcopy(rnn)  # Update best model
                        best_loss = smooth_loss  # Update best loss
                h_prev = h[:, -1]  # h_prev updated to the last computed hidden state
                e += seq_length  # Update the pointer
            if tweet_idx % 5000 == 0:
                x_0 = one_hot('\r', char_to_ind)
                samples = synthesize(best_rnn, h_prev, x_0, 140)
                samples = [ind_to_char[int(np.argmax(samples[:, n]))] for n in range(samples.shape[1])]
                print("\n")
                print("".join(samples))
                print("\n")
            if tweet_idx % 10 == 0:  # Append smooth loss each 10 tweets
                loss_list.append(smooth_loss)
            if tweet_idx % 100 == 0:
                print("Smooth loss: " + str(smooth_loss))
            e = 0  # Reset e when there are no enough characters
            h_prev = np.zeros(h_prev.shape)  # Reset h_prev to 0 after tweet

    plt.plot(np.arange(len(loss_list)) * 10, loss_list)
    plt.xlabel("Tweet")
    plt.ylabel("Loss")
    plt.show()

    x_0 = one_hot('\r', char_to_ind)
    print("Lowest loss: " + str(best_loss))
    h_prev = np.zeros(rnn.m)
    samples = synthesize(best_rnn, h_prev, x_0, 140)
    samples = [ind_to_char[int(np.argmax(samples[:, n]))] for n in range(samples.shape[1])]
    print("\n")
    print("".join(samples))
    print("\n")


if __name__ == "__main__":
    main()
