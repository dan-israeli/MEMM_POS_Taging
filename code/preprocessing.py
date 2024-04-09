from collections import OrderedDict, defaultdict
from typing import List, Dict, Tuple
from string import punctuation
from scipy import sparse
import numpy as np


DET_THRESHOLD = 500
WORD, TAG = 0, 1
SET, CNT = 0, 1


def extract_sentences_and_tags(file_path):
    sentences, tags = [], []

    with open(file_path) as file:
        for line in file:

            if line[-1:] == "\n":
                line = line[:-1]

            words_and_tags = line.split(' ')

            c_sentence, c_tags = [], []
            for word_and_tag in words_and_tags:
                c_word, c_tag = word_and_tag.split("_")
                c_sentence.append(c_word)
                c_tags.append(c_tag)

            sentences.append(c_sentence)
            tags.append(c_tags)

    return sentences, tags


def pad_sentences(sentences):
    padded_sentences = []

    for sentence in sentences:
        # adds start of sentence tokens
        padded_sentence = ["*", "*"]

        for word in sentence:
            padded_sentence.append(word)

        # adds end of sentence token
        padded_sentence.append("~")
        padded_sentences.append(padded_sentence)

    return padded_sentences


def find_symbol_tags(sentences, tags):
    """
    find all tags that do not contain ANY english letter (e.g. ".", "#", etc.)
    """
    symbol_tags = set()

    for c_sentence, c_tags in zip(sentences, tags):
        for c_word, c_tag in zip(c_sentence, c_tags):

            # c tag does not contain ANY english letters
            if str.lower(c_tag) == str.upper(c_tag):
                symbol_tags.add(c_tag)

    return symbol_tags


def find_possible_tags(sentences, tags, all_tags, symbol_tags):

    word_tags_set_dict = {}

    for c_sentence, c_tags in zip(sentences, tags):
        for c_word, c_tag in zip(c_sentence, c_tags):

            if c_word not in word_tags_set_dict:
                word_tags_set_dict[c_word] = set()

            word_tags_set_dict[c_word].add(c_tag)


    non_det_symbol_tags = set()

    for word_tags_set in word_tags_set_dict.values():

        if len(word_tags_set) > 1:

           for symbol_tag in symbol_tags:
                if symbol_tag in word_tags_set:
                    non_det_symbol_tags.add(symbol_tag)

    print(symbol_tags)
    print(non_det_symbol_tags)

    possible_tags = all_tags - symbol_tags | non_det_symbol_tags
    return possible_tags

def find_det_words(sentences, tags, symbol_tags):
    """
    find all words that we are sure they have only one possible tag.
    these words are all the words in the training set that have only one tag,
    and meet one of the following conditions:
    1. the only tag seen is a symbol tag
    2. the word has benn encountered at least THRESHOLD times
    """

    word_dict = {}
    det_words2tags = {}

    for c_sentence, c_tags in zip(sentences, tags):
        for c_word, c_tag in zip(c_sentence, c_tags):

            if c_word not in word_dict:
                word_dict[c_word] = [set([c_tag]), 1]

            else:
                word_dict[c_word][SET].add(c_tag)
                word_dict[c_word][CNT] += 1

    for word, (s, cnt) in word_dict.items():
        s = list(s)

        # the word has been seen with only one tag and
        # the tag is a symbol or the word has been uncounted at least THRESHOLD times
        if len(s) == 1 and (s[0] in symbol_tags or cnt >= DET_THRESHOLD):
            det_words2tags[word] = s[0]

    return set(det_words2tags.keys()), det_words2tags


def find_possible_tags_indices(tag2idx, possible_tags):
    possible_tags_indices = []
    for tag, idx in tag2idx.items():

        if tag in possible_tags:
            possible_tags_indices.append(idx)

    return possible_tags_indices


def increment_dict(dict, key):
    if key not in dict:
        dict[key] = 1

    else:
        dict[key] += 1


def contains_number(s):
    # return true if s contains at least one number
    return any(c.isdigit() for c in s)


def contains_punctuation(s):
    # return true if s contains at least one number
    return any(c in punctuation for c in s)


def find_word_frequency(word, sentence):
    freq = 0
    for word_in_sen in sentence:
        if word == word_in_sen:
            freq += 1

    return freq


class FeatureStatistics:
    def initialize_tag_index_dicts(self):
        # add '*' as a possible tag (needed for "memm_viterbi" function)
        all_tags = ['*'] + list(self.tags)

        for i, tag in enumerate(all_tags):
            self.tag2idx[tag] = i
            self.idx2tag[i] = tag

    def __init__(self, sentences, tags):
        # Total number of features accumulated
        self.n_total_features = 0

        # Init all features dictionaries
        # the feature classes used in the code
        self.feature_dict_list = [f"f{i}" for i in range(100, 108)] + ["f_characters", "f_num", "f_word_len", "f_position",
                                                                       "f_frequency", "f_question", "f_ctag_nword"]

        # a dictionary containing the counts of each data regarding a feature class
        self.feature_rep_dict = {fd: OrderedDict() for fd in self.feature_dict_list}

        self.tags = set()  # a set of all the seen tags
        self.symbol_tags = set() # a set of all the seen symbol tags (e.g. '#'. ':', etc.)
        self.possible_tags = set() # a set of possible tags for non-deterministic words

        self.tags_counts = defaultdict(int)  # a dictionary with the number of times each tag appeared in the text
        self.words_count = defaultdict(int)  # a dictionary with the number of times each word appeared in the text
        self.histories = []  # a list of all the histories seen at the train set

        self.det_words = set()
        self.det_words2tags = {}

        self.tag2idx = {}  # a dictionary which assign each tag an index (integer)
        self.idx2tag = {}  # a dictionary which assign the corresponding index of a tag to it

        self.possible_tags_indices = []

        self.initialize(sentences, tags)

    def initialize(self, sentences, tags):

        for c_sentence, c_tags in zip(sentences, tags):

            # add start of sentence tokens
            padded_c_sentence = ["*", "*"]
            padded_c_tags = ["*", "*"]

            for c_word, c_tag in zip(c_sentence, c_tags):

                self.tags.add(c_tag)
                self.tags_counts[c_tag] += 1
                self.words_count[c_word] += 1

                padded_c_sentence.append(c_word)
                padded_c_tags.append(c_tag)

            # add end of sentence tokens
            padded_c_sentence.append("~")
            padded_c_tags.append("~")

            # create histories
            n = len(padded_c_sentence)
            for i in range(2, n - 1):
                # tags
                c_tag, p_tag, pp_tag = padded_c_tags[i], padded_c_tags[i-1], padded_c_tags[i-2]

                # words
                c_word, p_word, pp_word = padded_c_sentence[i], padded_c_sentence[i-1], padded_c_sentence[i-2]
                next_word, last_word = padded_c_sentence[i+1], padded_c_sentence[n-2]

                # current word frequency in the sentence
                c_word_freq = find_word_frequency(c_word, padded_c_sentence)

                # create history
                history = (c_tag, p_tag, pp_tag, c_word, p_word, pp_word,
                           next_word, last_word, i, n, c_word_freq)

                self.histories.append(history)

        # after finding all the different tags in the train data,
        # we can initialize index_to_tag and tag_to_index dictionaries
        self.initialize_tag_index_dicts()

        # find the symbol tags and the deterministic words
        self.symbol_tags = find_symbol_tags(sentences, tags)
        # find the possible tags for non-deterministic words
        self.possible_tags = find_possible_tags(sentences, tags, self.tags, self.symbol_tags)
        # find the deterministic words
        self.det_words, self.det_words2tags = find_det_words(sentences, tags, self.symbol_tags)

        # find the possible tags indices (needed for "memm_viterbi" function)
        self.possible_tags_indices = find_possible_tags_indices(self.tag2idx, self.possible_tags)

        print(self.possible_tags)
        print(self.possible_tags_indices)

    def get_f_count(self) -> None:
        """
            Extract out of text all word/tag pairs
            @param: file_path: full path of the file to read
            Updates the histories list
        """
        for history in self.histories:
            c_tag, p_tag, pp_tag, c_word, p_word, pp_word, next_word, last_word, i, n, c_word_freq = history
            len_c_word = len(c_word)

            # f100
            increment_dict(self.feature_rep_dict["f100"], key=(c_tag, c_word))

            # f101 + f102
            for i in range(1, 5):

                # the length of the word is smaller than the prefix/suffix length
                if len_c_word < i:
                    break

                # f101
                c_word_suf = c_word[-i:]
                increment_dict(self.feature_rep_dict["f101"], key=(c_tag, c_word_suf))

                # f102
                c_word_pref = c_word[:i]
                increment_dict(self.feature_rep_dict["f102"], key=(c_tag, c_word_pref))

            # f103
            increment_dict(self.feature_rep_dict["f103"], key=(c_tag, p_tag, pp_tag))

            # f104
            increment_dict(self.feature_rep_dict["f104"], key=(c_tag, p_tag))

            # f105
            increment_dict(self.feature_rep_dict["f105"], key=c_tag)

            # f106
            if str.lower(p_word) == "the":
                increment_dict(self.feature_rep_dict["f106"], key=c_tag)

            # f107
            if str.lower(next_word) == "the":
                increment_dict(self.feature_rep_dict["f107"], key=c_tag)

            # f_characters

            # words that all letters are lowercase
            if c_word.islower():
                increment_dict(self.feature_rep_dict["f_characters"], key=(c_tag, "all_lower"))

            # words that all letters are uppercase
            if c_word.isupper():
                increment_dict(self.feature_rep_dict["f_characters"], key=(c_tag, "all_upper"))

            # words that contain a lowercase letter and an uppercase letter
            if (not c_word.islower()) and (not c_word.isupper()):
                increment_dict(self.feature_rep_dict["f_characters"], key=(c_tag, "lower_and_upper"))

            # words that contain a letter and a punctuation
            if (str.lower(c_word) != str.upper(c_word)) and (contains_punctuation(c_word)):
                increment_dict(self.feature_rep_dict["f_characters"], key=(c_tag, "letters_and_punctuation"))

            # words that only the first letter is uppercase
            if c_word[0].isupper() and (len(c_word) == 1 or c_word[1:].islower()):
                increment_dict(self.feature_rep_dict["f_characters"], key=(c_tag, "starts_upper"))

            # words that the last character is '.' (but the word is not '.')
            if str.lower(c_word) != str.upper(c_word) and c_word[-1] == '.':
                increment_dict(self.feature_rep_dict["f_characters"], key=(c_tag, "ends_period"))

            # f_num

            # check if the word contain at least one number
            if contains_number(c_word):

                # words that do not contain letters, therefore all characters are numbers (and maybe with some symbols)
                if str.lower(c_word) == str.upper(c_word):
                    increment_dict(self.feature_rep_dict["f_num"], key=(c_tag, "all_num"))

                # words that contain a number and a letter
                else:
                    increment_dict(self.feature_rep_dict["f_num"], key=(c_tag, "num_and_letter"))

            # current tag and word length
            increment_dict(self.feature_rep_dict["f_word_len"], key=(c_tag, len_c_word))

            # current tag and position
            # shift i two positions left since the sentence starts with ("*" "*")
            increment_dict(self.feature_rep_dict["f_position"], key=(c_tag, i-2))

            # current tag and relative position

            # shift i two positions left since the sentence starts with ("*" "*")
            # the true length of the sentence is n-3 since we add ("*" "*") at the start and ("~") at the end
            relative_position = (i - 2) / (n - 3)

            # current tag in start position
            if 0 <= relative_position < 1 / 3:
                increment_dict(self.feature_rep_dict["f_position"], key=(c_tag, "start"))

            # current tag in middle position
            elif 1 / 3 <= relative_position < 2 / 3:
                increment_dict(self.feature_rep_dict["f_position"], key=(c_tag, "middle"))

            # current tag in end position
            elif 2 / 3 <= relative_position <= 1:
                increment_dict(self.feature_rep_dict["f_position"], key=(c_tag, "end"))

            # current tag and current word frequency
            increment_dict(self.feature_rep_dict["f_frequency"], key=(c_tag, c_word_freq))

            # current tag and current sentence is a question
            if last_word == "?":
                increment_dict(self.feature_rep_dict["f_question"], key=c_tag)

            # current tag and next word
            increment_dict(self.feature_rep_dict["f_ctag_nword"], key=(c_tag, next_word))


class Feature2id:
    def __init__(self, feature_statistics: FeatureStatistics, threshold: int):
        """
        @param feature_statistics: the feature statistics object
        @param threshold: the minimal number of appearances a feature should have to be taken
        """
        self.feature_statistics = feature_statistics  # statistics class, for each feature gives empirical counts
        self.threshold = threshold  # feature count threshold - empirical count must be higher than this

        self.n_total_features = 0  # Total number of features accumulated

        # Init all features dictionaries (that we want to include in our model)
        print(feature_statistics.feature_dict_list)
        self.feature_to_idx = {
            feat_class: OrderedDict() for feat_class in feature_statistics.feature_dict_list
        }

        self.represent_input_with_features = OrderedDict()
        self.histories_matrix = OrderedDict()
        self.histories_features = OrderedDict()
        self.small_matrix = sparse.csr_matrix
        self.big_matrix = sparse.csr_matrix

    def get_features_idx(self) -> None:
        """
        Assigns each feature that appeared enough time in the train files an idx.
        Saves those indices to self.feature_to_idx
        """
        for feat_class in self.feature_statistics.feature_rep_dict:

            if feat_class not in self.feature_to_idx:
                continue

            for feat, count in self.feature_statistics.feature_rep_dict[feat_class].items():
                if count >= self.threshold:

                    self.feature_to_idx[feat_class][feat] = self.n_total_features
                    self.n_total_features += 1

        print(f"you have {self.n_total_features} features!")

        for dict_key in self.feature_to_idx:
            print(dict_key, len(self.feature_to_idx[dict_key]))

    def calc_represent_input_with_features(self) -> None:
        """
        initializes the matrices used in the optimization process - self.big_matrix and self.small_matrix
        """

        big_r = 0
        big_rows = []
        big_cols = []
        small_rows = []
        small_cols = []

        for small_r, hist in enumerate(self.feature_statistics.histories):
            for c in represent_input_with_features(hist, self.feature_to_idx):
                small_rows.append(small_r)
                small_cols.append(c)
            for r, y_tag in enumerate(self.feature_statistics.tags):
                demi_hist = tuple([y_tag]) + hist[1:]
                self.histories_features[demi_hist] = []
                for c in represent_input_with_features(demi_hist, self.feature_to_idx):
                    big_rows.append(big_r)
                    big_cols.append(c)
                    self.histories_features[demi_hist].append(c)
                big_r += 1
        self.big_matrix = sparse.csr_matrix((np.ones(len(big_rows)), (np.array(big_rows), np.array(big_cols))),
                                            shape=(len(self.feature_statistics.tags) * len(
                                                self.feature_statistics.histories), self.n_total_features),
                                            dtype=bool)
        self.small_matrix = sparse.csr_matrix(
            (np.ones(len(small_rows)), (np.array(small_rows), np.array(small_cols))),
            shape=(len(
                self.feature_statistics.histories), self.n_total_features), dtype=bool)


def represent_input_with_features(history: Tuple, dict_of_dicts: Dict[str, Dict[Tuple[str, str], int]]) -> List[int]:
    """
        Extract feature vector in per a given history
        @param history: tuple{c_word, c_tag, p_word, p_tag, pp_word, pp_tag, n_word}
        @param dict_of_dicts: a dictionary of each feature and the index it was given
        @return a list with all features that are relevant to the given history
    """
    c_tag, p_tag, pp_tag, c_word, p_word, pp_word, next_word, last_word, i, n, c_word_freq = history
    len_c_word = len(c_word)

    features = []

    # f100
    if (c_tag, c_word) in dict_of_dicts["f100"]:
        features.append(dict_of_dicts["f100"][(c_tag, c_word)])

    # f101 + f102
    for i in range(1, 5):

        # the length of the word is smaller than the prefix/suffix length
        if len_c_word < i:
            break

        # f101
        c_word_suf = c_word[-i:]
        if (c_tag, c_word_suf) in dict_of_dicts["f101"]:
            features.append(dict_of_dicts["f101"][(c_tag, c_word_suf)])

        # f102
        c_word_pref = c_word[:i]
        if (c_tag, c_word_pref) in dict_of_dicts["f102"]:
            features.append(dict_of_dicts["f102"][(c_tag, c_word_pref)])

    # f103
    if (c_tag, p_tag, pp_tag) in dict_of_dicts["f103"]:
        features.append(dict_of_dicts["f103"][(c_tag, p_tag, pp_tag)])

    # f104
    if (c_tag, p_tag) in dict_of_dicts["f104"]:
        features.append(dict_of_dicts["f104"][(c_tag, p_tag)])

    # f105
    if c_tag in dict_of_dicts["f105"]:
        features.append(dict_of_dicts["f105"][c_tag])

    # f106
    if (str.lower(p_word) == "the") and (c_tag in dict_of_dicts["f106"]):
        features.append(dict_of_dicts["f106"][c_tag])

    # f107
    if (str.lower(next_word) == "the") and (c_tag in dict_of_dicts["f107"]):
        features.append(dict_of_dicts["f107"][c_tag])

    # f_characters

    # words that all letters are lowercase
    if (c_word.islower()) and ((c_tag, "all_lower") in dict_of_dicts["f_characters"]):
        features.append(dict_of_dicts["f_characters"][(c_tag, "all_lower")])

    # words that all letters are uppercase
    if (c_word.isupper()) and ((c_tag, "all_upper") in dict_of_dicts["f_characters"]):
        features.append(dict_of_dicts["f_characters"][(c_tag, "all_upper")])

    # words that contain a lowercase letter and an uppercase letter
    if ((not c_word.islower()) and (not c_word.isupper())) and ((c_tag, "lower_and_upper") in dict_of_dicts["f_characters"]):
        features.append(dict_of_dicts["f_characters"][(c_tag, "lower_and_upper")])

    # words that contain a letter and a punctuation
    if ((str.lower(c_word) != str.upper(c_word)) and (contains_punctuation(c_word))) and ((c_tag, "letters_and_punctuation") in dict_of_dicts["f_characters"]):
        features.append(dict_of_dicts["f_characters"][(c_tag, "letters_and_punctuation")])

    # words that only the first letter is uppercase
    if (c_word[0].isupper() and (len(c_word) == 1 or c_word[1:].islower())) and ((c_tag, "starts_upper") in dict_of_dicts["f_characters"]):
        features.append(dict_of_dicts["f_characters"][(c_tag, "starts_upper")])

    # words that the last character in word is '.' (but the word is not '.')
    if (str.lower(c_word) != str.upper(c_word) and c_word[-1] == '.') and ((c_tag, "ends_period") in dict_of_dicts["f_characters"]):
        features.append(dict_of_dicts["f_characters"][(c_tag, "ends_period")])

    # f_num

    # check if the word contain at least one number
    if contains_number(c_word):
        # words that do not contain letters, therefore all characters are numbers (and maybe with some symbols)
        if (str.lower(c_word) == str.upper(c_word)) and ((c_tag, "all_num") in dict_of_dicts["f_num"]):
            features.append(dict_of_dicts["f_num"][(c_tag, "all_num")])

        # words that contain a number and a letter
        elif (c_tag, "num_and_letter") in dict_of_dicts["f_num"]:
            features.append(dict_of_dicts["f_num"][(c_tag, "num_and_letter")])

    # current tag and word length
    if (c_tag, len_c_word) in dict_of_dicts["f_word_len"]:
        features.append(dict_of_dicts["f_word_len"][(c_tag, len_c_word)])

    # current tag and current position
    # shift i two positions left since the sentence starts with ("*" "*")
    if (c_tag, i-2) in dict_of_dicts["f_position"]:
        features.append(dict_of_dicts["f_position"][(c_tag, i-2)])

    # current tag and relative position

    # shift i two positions left since the sentence starts with ("*" "*")
    # the true length of the sentence is n-3 since we add ("*" "*") at the start and ("~") at the end
    relative_position = (i-2) / (n-3)

    # current tag and start position
    if (0 <= relative_position < 1/3) and ((c_tag, "start") in dict_of_dicts["f_position"]):
        features.append(dict_of_dicts["f_position"][(c_tag, "start")])

    # current tag and middle position
    elif (1/3 <= relative_position < 2/3) and ((c_tag, "middle") in dict_of_dicts["f_position"]):
        features.append(dict_of_dicts["f_position"][(c_tag, "middle")])

    # current tag and end position
    elif (2/3 <= relative_position <= 1) and ((c_tag, "end") in dict_of_dicts["f_position"]):
        features.append(dict_of_dicts["f_position"][(c_tag, "end")])

    # current tag and current word frequency
    if (c_tag, c_word_freq) in dict_of_dicts["f_frequency"]:
        features.append(dict_of_dicts["f_frequency"][(c_tag, c_word_freq)])

    # current tag and current sentence is a question
    if (last_word == "?") and (c_tag in dict_of_dicts["f_question"]):
        features.append(dict_of_dicts["f_question"][c_tag])

    # current tag and next word
    if (c_tag, next_word) in dict_of_dicts["f_ctag_nword"]:
        features.append(dict_of_dicts["f_ctag_nword"][(c_tag, next_word)])

    return features


def preprocess_train(sentences, tags, threshold):

    # Statistics
    statistics = FeatureStatistics(sentences, tags)
    statistics.get_f_count()

    # feature2id
    feature2id = Feature2id(statistics, threshold)
    feature2id.get_features_idx()

    feature2id.calc_represent_input_with_features()

    return statistics, feature2id


def read_test(file_path, tagged=True) -> List[Tuple[List[str], List[str]]]:
    """
    reads a test file
    @param file_path: the path to the file
    @param tagged: whether the file is tagged (validation set) or not (test set)
    @return: a list of all the sentences, each sentence represented as tuple of list of the words and a list of tags
    """
    list_of_sentences = []
    with open(file_path) as f:
        for line in f:
            if line[-1:] == "\n":
                line = line[:-1]
            sentence = (["*", "*"], ["*", "*"])
            split_words = line.split(' ')
            for word_idx in range(len(split_words)):
                if tagged:
                    cur_word, cur_tag = split_words[word_idx].split('_')
                else:
                    cur_word, cur_tag = split_words[word_idx], ""
                sentence[WORD].append(cur_word)
                sentence[TAG].append(cur_tag)
            sentence[WORD].append("~")
            sentence[TAG].append("~")
            list_of_sentences.append(sentence)
    return list_of_sentences






