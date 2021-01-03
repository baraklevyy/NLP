import os
from data import *
from collections import defaultdict
from collections import Counter

def most_frequent_train(train_data):
    """
    Gets training data that includes tagged sentences.
    Returns a dictionary that maps every word in the training set to its most frequent tag.
    The dictionary should have a default value.
    """
    ### YOUR CODE HERE
    histogram = {}
    final_dict = []
    for sentence in train_data:
        for w_t_pair in sentence:
            if w_t_pair[0] not in histogram:
                histogram[w_t_pair[0]] = {w_t_pair[-1]: 1}
            else:
                inc = histogram[w_t_pair[0]].get(w_t_pair[-1], 0) + 1
                histogram[w_t_pair[0]][w_t_pair[-1]] = inc
    final_dict = [(w, max(t, key=t.get)) for w,t in histogram.items()]
    return dict(final_dict)
    ### END YOUR CODE

def most_frequent_eval(test_set, pred_tags):
    """
    Gets test data and tag prediction map.
    Returns an evaluation of the accuracy of the most frequent tagger.
    """
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_set:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)

        ### YOUR CODE HERE
        sent_pred = []
        for word in words:
            if word in pred_tags:
                sent_pred.append(pred_tags.get(word))
            else:
                sent_pred.append('PER') # some default tag
        pred_tag_seqs.append(sent_pred)
        ### END YOUR CODE

    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    train_sents = read_conll_ner_file("/content/drive/My Drive/NLP_HW3/data/train.conll")
    dev_sents = read_conll_ner_file("/content/drive/My Drive/NLP_HW3/data/dev.conll")
    vocab = compute_vocab_count(train_sents)
    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)
    
  


    model = most_frequent_train(train_sents)
    most_frequent_eval(dev_sents, model)

