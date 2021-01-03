import os
import random
import time

import numpy as np
import pandas as pd

from data_utils import utils
from sgd import sgd
from q1c_neural import forward, forward_backward_prop


VOCAB_EMBEDDING_PATH = "data/lm/vocab.embeddings.glove.txt"
BATCH_SIZE = 50
NUM_OF_SGD_ITERATIONS = 40000
LEARNING_RATE = 0.3


def load_vocab_embeddings(path=VOCAB_EMBEDDING_PATH):
    result = []
    with open(path) as f:
        index = 0
        for line in f:
            line = line.strip()
            row = line.split()
            data = [float(x) for x in row[1:]]
            assert len(data) == 50
            result.append(data)
            index += 1
    return result


def load_data_as_sentences(path, word_to_num):
    """
    Converts the training data to an array of integer arrays.
      args: 
        path: string pointing to the training data
        word_to_num: A dictionary from string words to integers
      returns:
        An array of integer arrays. Each array is a sentence and each 
        integer is a word.
    """
    docs_data = utils.load_dataset(path)
    S_data = utils.docs_to_indices(docs_data, word_to_num)
    return docs_data, S_data


def convert_to_lm_dataset(S):
    """
    Takes a dataset that is a list of sentences as an array of integer arrays.
    Returns the dataset a bigram prediction problem. For any word, predict the
    next work. 
    IMPORTANT: we have two padding tokens at the beginning but since we are 
    training a bigram model, only one will be used.
    """
    in_word_index, out_word_index = [], []
    for sentence in S:
        for i in range(len(sentence)):
            if i < 2:
                continue
            in_word_index.append(sentence[i - 1])
            out_word_index.append(sentence[i])
    return in_word_index, out_word_index


def shuffle_training_data(in_word_index, out_word_index):
    combined = list(zip(in_word_index, out_word_index))
    random.shuffle(combined)
    return list(zip(*combined))


def int_to_one_hot(number, dim):
    res = np.zeros(dim)
    res[number] = 1.0
    return res


def lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, params):
    """
    Takes a dataset that is a list of sentences as an array of integer arrays.
        args:
        in_word_index, out_word_index: convert_to_lm_dataset() outputs. 2 python lists of words and matching prediction.
        num_to_word_embedding: GloVe word embeddings
      returns:
        An array of integer arrays. Each array is a sentence and each
        integer is a word.
    """
    data = np.zeros([BATCH_SIZE, input_dim])
    labels = np.zeros([BATCH_SIZE, output_dim])

    # Construct the data batch and run your backpropogation implementation
    ### YOUR CODE HERE

    rand_idxs = np.random.choice(len(in_word_index), size=BATCH_SIZE)    # np array with batch_size random idxs

    for batch_idx in range(BATCH_SIZE):
        rand_idx = rand_idxs[batch_idx]

        # data
        rand_word_in_idx = in_word_index[rand_idx]                          # get a random word idx from the dataset
        rand_word_in_embedding = num_to_word_embedding[rand_word_in_idx]    # get embedding for the random word chosen
        data[batch_idx, :] = rand_word_in_embedding

        # target
        rand_word_out_idx = out_word_index[rand_idx]    # get corresponding out word
        labels[batch_idx, rand_word_out_idx] = 1        # word idx to one-hot vector

        cost, grad = forward_backward_prop(data, labels, params, dimensions)


    ### END YOUR CODE

    cost /= BATCH_SIZE
    grad /= BATCH_SIZE
    return cost, grad


def eval_neural_lm(eval_data_path):
    """
    Evaluate perplexity (use dev set when tuning and test at the end)
    """
    _, S_dev = load_data_as_sentences(eval_data_path, word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_dev)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    perplexity = 0

    ### YOUR CODE HERE

    log_prob = 0
    # iterate dev set and sum up the log prob for perplexity calculation
    for idx in range(num_of_examples):
        data = num_to_word_embedding[in_word_index[idx]]    # word embedding vec
        target = out_word_index[idx]                        # corresponding target idx

        # forward returns the probability calculated for the target word
        target_prob = forward(data, target, params, dimensions)
        log_prob += np.log2(target_prob)    # equivalent to sigma(log(q(x_I))

    perplexity = 2**(-log_prob/num_of_examples)

    ### END YOUR CODE

    return perplexity


if __name__ == "__main__":
    # Load the vocabulary
    vocab = pd.read_table("data/lm/vocab.ptb.txt",
                          header=None, sep="\s+", index_col=0, names=['count', 'freq'], )

    vocabsize = 2000
    num_to_word = dict(enumerate(vocab.index[:vocabsize]))
    num_to_word_embedding = load_vocab_embeddings()
    word_to_num = utils.invert_dict(num_to_word)

    # Load the training data
    _, S_train = load_data_as_sentences('data/lm/ptb-train.txt', word_to_num)
    in_word_index, out_word_index = convert_to_lm_dataset(S_train)
    assert len(in_word_index) == len(out_word_index)
    num_of_examples = len(in_word_index)

    random.seed(31415)
    np.random.seed(9265)
    in_word_index, out_word_index = shuffle_training_data(in_word_index, out_word_index)
    startTime = time.time()

    # Training should happen here
    # Initialize parameters randomly
    # Construct the params
    input_dim = 50
    hidden_dim = 50
    output_dim = vocabsize
    dimensions = [input_dim, hidden_dim, output_dim]
    params = np.random.randn((input_dim + 1) * hidden_dim + (
        hidden_dim + 1) * output_dim, )
    print(f"#params: {len(params)}")
    print(f"#train examples: {num_of_examples}")

    # run SGD
    params = sgd(
            lambda vec: lm_wrapper(in_word_index, out_word_index, num_to_word_embedding, dimensions, vec),
            params, LEARNING_RATE, NUM_OF_SGD_ITERATIONS, None, True, 1000)

    print(f"training took {time.time() - startTime} seconds")

    # Evaluate perplexity with dev-data
    perplexity = eval_neural_lm('data/lm/ptb-dev.txt')
    print(f"dev perplexity : {perplexity}")

    # Evaluate perplexity with test-data (only at test time!)
    if os.path.exists('data/lm/ptb-test.txt'):
        perplexity = eval_neural_lm('data/lm/ptb-test.txt')
        print(f"test perplexity : {perplexity}")
    else:
        print("test perplexity will be evaluated only at test time!")
