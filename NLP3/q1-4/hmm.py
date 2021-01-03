import os
import time
import numpy as np
from data import *
from collections import defaultdict, Counter

def hmm_train(sents):
    """
        sents: list of tagged sentences
        Returns: the q-counts and e-counts of the sentences' tags, total number of tokens in the sentences
    """

    print("Start training")
    total_tokens = 0
    # YOU MAY OVERWRITE THE TYPES FOR THE VARIABLES BELOW IN ANY WAY YOU SEE FIT
    q_tri_counts, q_bi_counts, q_uni_counts, e_tag_counts = [defaultdict(int) for i in range(4)]
    e_word_tag_counts = defaultdict(lambda: defaultdict(int))
    ### YOUR CODE HERE

    ###############################
    for sent in sents:
        for pair in sent:
            e_word_tag_counts[pair[0]][pair[1]] += 1
            e_tag_counts[pair[1]] += 1
    ####################################################
    total_tokens = sum([len(sent) for sent in sents])
    #################################################
    sents_only_tags = [[word[1] for word in sent] for sent in sents]
    for sent in sents_only_tags:
        sent.append('STOP')
        sent.insert(0, '*')
        sent.insert(0, '*')
        sent.insert(0, '*')
    ###############################################
    for sent in sents_only_tags:
        for tag in sent:
            q_uni_counts[tag] += 1
    ###############################################
    for sent in sents_only_tags:
        for i,tag in enumerate(sent):
            if tag != 'STOP':
                q_bi_counts[(tag,sent[i+1])] += 1
    ###############################################
    for sent in sents_only_tags:
        for i in range(len(sent)):
            if i + 2 < len(sent):
                q_tri_counts[(sent[i], sent[i+1], sent[i+2])] += 1
    #################################################        



    ### END YOUR CODE
    return total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts


def hmm_viterbi(sent, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts,
                e_word_tag_counts, e_tag_counts, lambda1, lambda2):
    """
        Receives: a sentence to tag and the parameters learned by hmm
        Returns: predicted tags for the sentence
    """
    predicted_tags = ["O"] * (len(sent))
    ### YOUR CODE HERE
    ###################################################################
    ############## UTIL ###############################################
    def log_div(A,B):
        if (A == 0) or (B == 0):
            return 0
        else:
            return A/B
    ###################################################################
    ###################################################################
    all_tags = [*q_uni_counts]
    q_interpolated = defaultdict(lambda: defaultdict(int))
    e_conditioned = defaultdict(lambda: defaultdict(int))
    for tag1 in all_tags:
        for tag2 in all_tags:
            for tag3 in all_tags:
                q_interpolated[tag1][(tag3,tag2)] = lambda1*(log_div(q_tri_counts[(tag3,tag2,tag1)],q_bi_counts[(tag3, tag2)]))+lambda2*(log_div(q_bi_counts[(tag2,tag1)],q_uni_counts[tag2])) +(1-lambda2-lambda1)*(log_div(q_uni_counts[tag1],total_tokens))

    
##################################################################
    super_s = [*e_tag_counts] 
    s_sets = [super_s]*(len(sent))
#################### Pruning ##################################
    for i, word in enumerate(sent):
        if bool(e_word_tag_counts[word]):
            s_sets[i] = [*e_word_tag_counts[word]]
################################################################
    s_sets.insert(0, ['*'])
    s_sets.insert(0, ['*'])
    pi_vals = defaultdict(lambda: defaultdict(int))
    bp_vals = defaultdict(lambda: defaultdict(int))
    pi_vals[0][('*','*')] = 1

    for k in range(1,len(sent)+1):
        for u in s_sets[(k-1)+1]:
            for v in s_sets[(k)+1]:
                max = -np.infty
                bp_max = 'NON'
                for w in s_sets[(k-2)+1]:
                    e_temp = log_div((e_word_tag_counts[sent[k-1]][v]),e_tag_counts[v])
                    if e_temp == 0:
                        e_temp = 1
                    temp = pi_vals[k-1][(w,u)]*q_interpolated[v][(w,u)]*e_temp
                    if max < temp:
                        max=temp
                        bp_max=w
                pi_vals[k][(u,v)] = max
                bp_vals[k][(u,v)] = bp_max
    #########################################################################
    max_u = 'O'
    max_v = 'O'
    max_val = -np.inf
    for u in all_tags:
        for v in all_tags:
            temp = pi_vals[len(sent)][(u,v)]*q_interpolated['STOP'][(u,v)]
            if max_val < temp:
                max_val = temp
                max_u = u
                max_v = v
    predicted_tags[-1] = max_v
    
    
    if len(sent)>2:
          predicted_tags[-2] = max_u
          for k in range(1,len(sent)-2+1):
            eff_k =len(sent) - 2 - k + 1
            predicted_tags[eff_k-1] = bp_vals[eff_k+2][(predicted_tags[eff_k+1-1], predicted_tags[eff_k+2-1])]
############################################################################################################################
    ### END YOUR CODE
    if 0 in predicted_tags:
        print("\n pred_tags:", predicted_tags)
        print("\n sent:", sent)
        print("\n pi_vals:", pi_vals)
    return predicted_tags

def hmm_eval(test_data, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts):
    """
    Receives: test data set and the parameters learned by hmm
    Returns an evaluation of the accuracy of hmm
    """
    print("Start evaluation")
    gold_tag_seqs = []
    pred_tag_seqs = []
    for sent in test_data:
        words, true_tags = zip(*sent)
        gold_tag_seqs.append(true_tags)
        ### YOUR CODE HERE
        temp = hmm_viterbi(list(words),total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts, 0.47, 0.37)
        pred_tag_seqs.append(tuple(temp))

        ### END YOUR CODE
  
    return evaluate_ner(gold_tag_seqs, pred_tag_seqs)

if __name__ == "__main__":
    start_time = time.time()
    train_sents = read_conll_ner_file("/content/drive/My Drive/NLP_HW3/data/train.conll")
    dev_sents = read_conll_ner_file("/content/drive/My Drive/NLP_HW3/data/dev.conll")
    vocab = compute_vocab_count(train_sents)

    train_sents = preprocess_sent(vocab, train_sents)
    dev_sents = preprocess_sent(vocab, dev_sents)

    total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts = hmm_train(train_sents)


    hmm_eval(dev_sents, total_tokens, q_tri_counts, q_bi_counts, q_uni_counts, e_word_tag_counts, e_tag_counts)
    train_dev_end_time = time.time()
    print("Train and dev evaluation elapsed: " + str(train_dev_end_time - start_time) + " seconds")
