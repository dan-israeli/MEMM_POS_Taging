from preprocessing import read_test, represent_input_with_features, find_word_frequency
from scipy.sparse import csr_matrix
from tqdm import tqdm
import numpy as np
import heapq

VAL, C_TAG, P_TAG = 0, 1, 2


def update_linear_terms_dict(demi_history_lst, linear_terms_vec):
    for demi_history, linear_term in zip(demi_history_lst, linear_terms_vec):
        LINEAR_TERM_DICT[demi_history] = linear_term


def update_normalization_term_dict(history, feature2id, v):
    possible_tags = feature2id.feature_statistics.possible_tags

    demi_history_lst, rows, cols = [], [], []
    row_num, non_zero_elements = 0, 0

    for y_tag in possible_tags:
        demi_history = tuple([y_tag]) + history[1:]
        demi_history_lst.append(demi_history)

        for c in represent_input_with_features(demi_history, feature2id.feature_to_idx):
            rows.append(row_num)
            cols.append(c)
            non_zero_elements += 1

        row_num += 1

    a = csr_matrix((np.ones(non_zero_elements, dtype=int), (rows, cols)), shape=(row_num, feature2id.n_total_features))
    linear_terms_vec = a.dot(v)

    # update the linear term dictionary with the linear terms that have been calculated in this function
    update_linear_terms_dict(demi_history_lst, linear_terms_vec)

    normalization_term = np.sum(np.e**linear_terms_vec)

    NORMALIZATION_TERM_DICT[history[1:]] = normalization_term


def q(history, feature2id, v, det_word_flag):

    # the word has only possible tag, therefore the q value is 1
    if det_word_flag:
        return 1

    if history[1:] not in NORMALIZATION_TERM_DICT:
        # update the normalization term dictionary with the normalization term of the given history
        update_normalization_term_dict(history, feature2id, v)

    normalization_term = NORMALIZATION_TERM_DICT[history[1:]]
    exp_linear_term = np.e ** LINEAR_TERM_DICT[history]

    return exp_linear_term / normalization_term


def insert_to_max_B(c_tup, max_pp_tag, bp_i, max_B_lst, len_max_B_lst, max_B_lst_full, B):

    # the list is not full (does not contain B elements) or the current max value is bigger than the minimum
    if (not max_B_lst_full) or (c_tup[VAL] > max_B_lst[0][VAL]):

        heapq.heappush(max_B_lst, c_tup)
        len_max_B_lst += 1

        bp_i[c_tup[C_TAG], c_tup[P_TAG]] = max_pp_tag

    # there are more than B elements, remove the minimum
    if len_max_B_lst > B:
        _, removed_c_tag, removed_p_tag = heapq.heappop(max_B_lst)
        len_max_B_lst -= 1

        del bp_i[(removed_c_tag, removed_p_tag)]

    if len_max_B_lst == B:
        max_B_lst_full = True

    return len_max_B_lst, max_B_lst_full


def split_max_B(max_B):
    """
    split 'max_B' into three lists: rows, cols, vals
    These lists will be used in order to create the current matrix (sparse matrix)
    """
    rows = [x[P_TAG] for x in max_B]
    cols = [x[C_TAG] for x in max_B]
    vals = [x[VAL] for x in max_B]

    return rows, cols, vals


def memm_viterbi(sentence, pre_trained_weights, feature2id, B):
    """
    Write your MEMM Viterbi implementation below
    You can implement Beam Search to improve runtime
    Implement q efficiently (refer to conditional probability definition in MEMM slides)
    """

    global NORMALIZATION_TERM_DICT, LINEAR_TERM_DICT
    LINEAR_TERM_DICT, NORMALIZATION_TERM_DICT = {}, {}

    global TAG2IDX, IDX2TAG
    TAG2IDX = feature2id.feature_statistics.tag2idx
    IDX2TAG = feature2id.feature_statistics.idx2tag

    all_tags = feature2id.feature_statistics.tags
    possible_tags_indices = feature2id.feature_statistics.possible_tags_indices
    det_words = feature2id.feature_statistics.det_words
    det_words2tags = feature2id.feature_statistics.det_words2tags

    tag_num = len(all_tags)
    n = len(sentence)

    # initialize the base case matrix
    # the shape is (tag_num+1, tag_num+1) to include "*" tag
    cur_rows, cur_cols = [TAG2IDX['*']], [TAG2IDX['*']]
    cur_matrix = csr_matrix(([1], (cur_rows, cur_cols)), shape=(tag_num+1, tag_num+1))

    # initialize the back-pointers list of dictionaries
    bp = [{} for i in range(n - 1)]

    # start from word in index 2 because the words in indices 0, 1 are '*'
    # in addition, end in index n - 2 (inclusive) because the word in index n - 1 is '~'
    for i in range(2, n - 1):
        prev_matrix = cur_matrix

        # the possible p tags are all the unique non-zero column indices of the previous matrix
        possible_p_tags = set(prev_matrix.indices)

        # list with elements in the form (c_tag, p_tag, val) where val is the corresponding max value of c_tag and p_tag
        # the list contain the top B elements w.r.t. val
        max_B_lst = []

        # these variables will be used in 'insert_to_max_B' function
        len_max_B_lst, max_B_lst_full = 0, False

        # turn max_B into a min-heap
        heapq.heapify(max_B_lst)

        cur_word = sentence[i]
        det_word_flag = False

        for c_tag in possible_tags_indices:

            if cur_word in det_words:
                # the word's tag is deterministic, no need to check any other tags
                c_tag_str = det_words2tags[cur_word]
                c_tag = TAG2IDX[c_tag_str]
                det_word_flag = True

            for p_tag in possible_p_tags:

                # the possible pp tags of a specific p tag are all the non-zero row indices
                possible_pp_tags = prev_matrix[:, p_tag].nonzero()[0]

                # get all the non-zero elements from the p tag column (in the previous matrix)
                prev_max_vals = prev_matrix.getcol(p_tag).data

                max_val, max_pp_tag = float('-inf'), 0
                for pp_tag, prev_max_val in zip(possible_pp_tags, prev_max_vals):

                    # initialize all the variables need to create the current history

                    # tags
                    c_tag_str, p_tag_str, pp_tag_str = IDX2TAG[c_tag], IDX2TAG[p_tag], IDX2TAG[pp_tag]

                    # words
                    c_word, p_word, pp_word = sentence[i], sentence[i-1], sentence[i-2]
                    next_word, last_word = sentence[i+1], sentence[n-2]

                    # current word frequency in the sentence
                    c_word_freq = find_word_frequency(sentence[i], sentence)

                    # create the current history
                    cur_history = (c_tag_str, p_tag_str, pp_tag_str, c_word, p_word, pp_word,
                                   next_word, last_word, i, n, c_word_freq)

                    # find the value of the current history
                    cur_val = prev_max_val * q(cur_history, feature2id, pre_trained_weights, det_word_flag)

                    # maintain 'max_val' and 'max_pp_tag' variables
                    if cur_val > max_val:
                        max_val = cur_val
                        max_pp_tag = pp_tag

                c_tup = (max_val, c_tag, p_tag)
                len_max_B_lst, max_B_lst_full = insert_to_max_B(c_tup, max_pp_tag, bp[i], max_B_lst, len_max_B_lst, max_B_lst_full, B)

            # c tag is deterministic, no need to check all other possibilities
            if det_word_flag:
                break

        # initialize the current matrix with the top B values
        cur_rows, cur_cols, cur_vals = split_max_B(max_B_lst)
        cur_matrix = csr_matrix((cur_vals, (cur_rows, cur_cols)), shape=(tag_num+1, tag_num+1))

    # get the index of the maximum value
    max_idx = np.argmax(np.array(cur_vals))

    # get the c tag and p tag which achieved the maximum value
    last_c_tag, last_p_tag = cur_cols[max_idx], cur_rows[max_idx]
    last_c_tag_str, last_p_tag_str = IDX2TAG[last_c_tag], IDX2TAG[last_p_tag]
    pred_tags = [last_c_tag_str]

    # check if the previous tag is not start of sentence token (happens for sentences of length 1)
    if last_p_tag_str != "*":
        pred_tags.append(last_p_tag_str)

    for i in range(n-2, 3, -1):
        last_pp_tag = bp[i][(last_c_tag, last_p_tag)]
        pred_tags.append(IDX2TAG[last_pp_tag])

        last_c_tag = last_p_tag
        last_p_tag = last_pp_tag

    # reverse the order of the list, tags were appended backwards
    pred_tags.reverse()
    return pred_tags


def tag_test_file(optimal_weights, feature2id, test_path, pred_path, B, tagged=True):

    test = read_test(test_path, tagged)

    output_file = open(pred_path, "w")

    for k, sen in tqdm(enumerate(test), total=len(test)):
        sentence = sen[0]
        pred = memm_viterbi(tuple(sentence), optimal_weights, feature2id, B)
        sentence = sentence[2:]

        for i in range(len(pred)):

            if i > 0:
                output_file.write(" ")

            output_file.write(f"{sentence[i]}_{pred[i]}")

        output_file.write("\n")

    output_file.close()
