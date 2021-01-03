from data import *
from sklearn.feature_extraction import DictVectorizer
from sklearn import linear_model
import time
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def build_extra_decoding_arguments(train_sents):
    """
    Receives: all sentences from training set
    Returns: all extra arguments which your decoding procedures requires
    """
    extra_decoding_arguments = {}
    ### YOUR CODE HERE
    # here we create the dictionary S which holds all possible tags for a word, seen in the training set
    # in this way we optimize the code as requested
    extra_decoding_arguments['S'] = dict()
    for sent in train_sents:
        for word_idx in range(len(sent)):
            word = sent[word_idx][0]
            tag = sent[word_idx][1]
            if word not in extra_decoding_arguments['S']:
                # init as set to get only one instance from a tag
                extra_decoding_arguments['S'][word] = set()
            extra_decoding_arguments['S'][word].add(tag)

    ### END YOUR CODE

    return extra_decoding_arguments


def extract_features_base(curr_word, next_word, prev_word, prevprev_word, prev_tag, prevprev_tag):
    """
        Receives: a word's local information
        Returns: The word's features.
    """
    features = {}
    features['word'] = curr_word

    ### YOUR CODE HERE
    # Ratnaparkhi features:
    # prefixes<5
    for i in range(1, min(5, len(curr_word))):
        features[f'prefix_{i}'] = curr_word[:i]
    # suffix<5
    for i in range(1, min(5, len(curr_word))):
        features[f'suffix_{i}'] = curr_word[:i]
    # contain number
    features[f'contain_digits'] = any(char.isdigit() for char in curr_word)
    # contain uppercase
    features[f'contain_digits'] = any(char.isupper() for char in curr_word)
    # contain uppercase
    features[f'contains_hyphen'] = 1 if '-' in curr_word else 0
    # prev tag
    features['prev_tag'] = prev_tag
    # prev_tag_prev_prev_tag
    features['prev_tag_prev_prev_tag'] = f"{prev_tag} {prevprev_tag}"
    # prev word
    features['prev_word'] = prev_word
    # prev prev word
    features['prev_prev_word'] = prevprev_word
    # next word
    features['next_word'] = next_word

    # additional features
    features['all_lower'] = curr_word.islower()
    features['all_upper'] = curr_word.isupper()
    features['length'] = len(curr_word)
    features['prev_prev_tag'] = prevprev_tag
    features['prev_word_prev_tag'] = f"{prev_word} {prev_tag}"
    features['prevprev_word_prevprev_tag'] = f"{prevprev_word} {prevprev_tag}"

    ### END YOUR CODE

    return features

def extract_features(sentence, i):
    curr_word = sentence[i][0]
    prev_token = sentence[i - 1] if i > 0 else ('<st>', '*')
    prevprev_token = sentence[i - 2] if i > 1 else ('<st>', '*')
    next_token = sentence[i + 1] if i < (len(sentence) - 1) else ('</s>', 'STOP')
    return extract_features_base(curr_word, next_token[0], prev_token[0], prevprev_token[0], prev_token[1], prevprev_token[1])

def vectorize_features(vec, features):
    """
        Receives: feature dictionary
        Returns: feature vector

        Note: use this function only if you chose to use the sklearn solver!
        This function prepares the feature vector for the sklearn solver,
        use it for tags prediction.
    """
    example = [features]
    return vec.transform(example)

def create_examples(sents, tag_to_idx_dict):
    examples = []
    labels = []
    num_of_sents = 0
    for sent in sents:
        num_of_sents += 1
        for i in range(len(sent)):
            features = extract_features(sent, i)
            examples.append(features)
            labels.append(tag_to_idx_dict[sent[i][1]])

    return examples, labels


def memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    for word_idx in range(len(sent)):
        features = extract_features(sent, word_idx)
        features_vec = vectorize_features(vec, features)
        pred = logreg.predict(features_vec)
        predicted_tags[word_idx] = index_to_tag_dict[pred[0]]
    ### END YOUR CODE
    return predicted_tags

def memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
        Receives: a sentence to tag and the parameters learned by memm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    def calc_q(sent, vec, k, u, v, t):
        featurs = extract_features(sent, k)
        featurs['prev_tag'] = u
        featurs['prev_prev_tag'] = t
        featurs['word'] = sent[k][0]
        v_features = vectorize_features(vec, featurs)
        pred = logreg.predict_proba(v_features).flatten()
        idx = tag_to_idx_dict[v]
        return pred[idx]


    # init tables as dictionaries
    pi = dict()     # pi (k,u,v)
    bp = dict()     # bp (k,u,v)
    S = dict()  # tags set was saved during training
    saved_word_tags = extra_decoding_arguments['S']
    n = len(sent)   # len of sentence
    S[-1] = {'*'}   # 2 first tags (-1, 0) are defined as tags
    S[-2] = {'*'}   # we use -1, -2 due to indexing conventions

    # base case as defined in class
    pi[(-1, '*', '*')] = 1.0
    for k in range(n):      # for k in {1..n} <=> {0,n-1}
        # init current S[k]
        # optimization: S is a saved dict with all tags we have seen to a given word during training
        # if recieved word is new, we calculate and save only the most probable tag for the given word.
        # we have seen that we don't loss much i accuracy compared to significant computational savings.
        curr_word = sent[k][0]
        if curr_word in saved_word_tags.keys():
            S[k] = saved_word_tags[curr_word]

        else:
            features = extract_features(sent, k)
            v_features = vectorize_features(vec, features)
            tags_prob_vec = logreg.predict_proba(v_features).flatten()
            S[k] = index_to_tag_dict[tags_prob_vec.argmax()]

        # viterbi algorithm
        end_max_prob = float('-Inf')
        for u in S[k-1]:  # for u in S[k-1]
            for v in S[k]:  # for v in S[k]
                max_prob = float('-inf')    # initial prob -inf
                for t in S[k-2]:    # find max in t \in (s[k-2])
                    prev_prob = pi[(k-1, t, u)]       # prob is given by pi(k-1,t,u)*q(v|t,u,w,k), need to find t that maximize this prob

                    # only calc prob that are high enough
                    # conditioned on the keys to verify that we don't get all tags into the pi dict
                    if prev_prob < 0.01 and (k, u, v) in pi.keys():
                        continue
                    q = calc_q(sent, vec, k, u, v, t)   # q func as defined in class
                    prob = prev_prob * q
                    if prob > max_prob:             # when ever we get better prob -> update
                        max_prob = prob             # update prob
                        pi[(k, u, v)] = prob        # pi(k,u,v) = pi(k-1,t_max,u)*q(v|t_max,u,w,k)
                        bp[(k, u, v)] = t           # bp(k,u,v) = argmax_t {pi(k-1,t_max,u)*q(v|t_max,u,w,k)}

            # last iteration, need to find u with max prob to choose y[n], y[n-1]
            if k == n-1 and pi[(k, u, v)] > end_max_prob:
                end_max_prob = pi[(k, u, v)]
                yn = v  # y[n] in class presentation
                ym = u  # # y[n-1] n class presentation

    predicted_tags[n-1] = yn    # last tag
    if n > 1:
        predicted_tags[n-2] = ym    # prev last tag
        # rest tags: for k=(n-2)...1 : y_k = bp(k+2, y_k+1, y_k+2)
        if n > 2:
            for j in range(n-3, -1, -1):
                predicted_tags[j] = bp[(j+2, predicted_tags[j+1], predicted_tags[j+2])]

    ### END YOUR CODE

    return predicted_tags


def memm_eval(test_data, logreg, vec, index_to_tag_dict, extra_decoding_arguments):
    """
    Receives: test data set and the parameters learned by memm
    Returns an evaluation of the accuracy of Viterbi & greedy memm
    """
    acc_viterbi, acc_greedy = 0.0, 0.0
    eval_start_timer = time.time()

    correct_greedy_preds = 0
    correct_viterbi_preds = 0
    total_words_count = 0

    gold_tag_seqs = []
    greedy_pred_tag_seqs = []
    viterbi_pred_tag_seqs = []
    wrong_preds = []
    for sent in tqdm(test_data):
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        #predict
        greedy_preds = memm_greedy(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        greedy_pred_tag_seqs.append(greedy_preds)
        viterbi_preds = memm_viterbi(sent, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
        viterbi_pred_tag_seqs.append(viterbi_preds)

        # updated accuracy variables
        true_tags_arr = np.asarray(true_tags)
        total_words_count += len(true_tags_arr)
        correct_greedy_preds += np.count_nonzero(true_tags_arr == np.asarray(greedy_preds))
        correct_viterbi_preds += np.count_nonzero(true_tags_arr == np.asarray(viterbi_preds))

        ## uncomment for saving t
        #not_equal_mask = np.in1d(true_tags, viterbi_preds)
        #for i in range(len(true_tags)):
        #    if not_equal_mask[i]:
        #        wrong_preds.append({"word": words[i], "true_tag": true_tags[i], "pred_tag": viterbi_preds[i]})


    acc_viterbi = correct_viterbi_preds/total_words_count
    acc_greedy = correct_greedy_preds/total_words_count
        ### END YOUR CODE

    greedy_evaluation = evaluate_ner(gold_tag_seqs, greedy_pred_tag_seqs)
    viterbi_evaluation = evaluate_ner(gold_tag_seqs, viterbi_pred_tag_seqs)

    return greedy_evaluation, viterbi_evaluation

def build_tag_to_idx_dict(train_sentences):
    curr_tag_index = 0
    tag_to_idx_dict = {}
    for train_sent in train_sentences:
        for token in train_sent:
            tag = token[1]
            if tag not in tag_to_idx_dict:
                tag_to_idx_dict[tag] = curr_tag_index
                curr_tag_index += 1

    tag_to_idx_dict['*'] = curr_tag_index
    return tag_to_idx_dict


if __name__ == "__main__":
    full_flow_start = time.time()
    train_sents = read_conll_ner_file("data/train.conll")
    dev_sents = read_conll_ner_file("data/dev.conll")

    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    extra_decoding_arguments = build_extra_decoding_arguments(train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    tag_to_idx_dict = build_tag_to_idx_dict(train_sents)
    index_to_tag_dict = invert_dict(tag_to_idx_dict)

    vec = DictVectorizer()
    print("Create train examples")
    train_examples, train_labels = create_examples(train_sents, tag_to_idx_dict)


    num_train_examples = len(train_examples)
    print("#example: " + str(num_train_examples))
    print("Done")

    print("Create dev examples")
    dev_examples, dev_labels = create_examples(dev_sents, tag_to_idx_dict)
    num_dev_examples = len(dev_examples)
    print("#example: " + str(num_dev_examples))
    print("Done")

    all_examples = train_examples
    all_examples.extend(dev_examples)

    print("Vectorize examples")
    all_examples_vectorized = vec.fit_transform(all_examples)
    train_examples_vectorized = all_examples_vectorized[:num_train_examples]
    dev_examples_vectorized = all_examples_vectorized[num_train_examples:]
    print("Done")

    logreg = linear_model.LogisticRegression(
        multi_class='multinomial', max_iter=128, solver='lbfgs', C=100000, verbose=1)
    print("Fitting...")
    start = time.time()
    logreg.fit(train_examples_vectorized, train_labels)
    end = time.time()
    print("End training, elapsed " + str(end - start) + " seconds")
    # End of log linear model training

    # Evaluation code - do not make any changes
    start = time.time()
    print("Start evaluation on dev set")

    memm_eval(dev_sents, logreg, vec, index_to_tag_dict, extra_decoding_arguments)
    end = time.time()

    print("Evaluation on dev set elapsed: " + str(end - start) + " seconds")
