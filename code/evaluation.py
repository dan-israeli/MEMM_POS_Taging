from preprocessing import extract_sentences_and_tags
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np


def extract_tags_list(tags):
    tags_lst = []

    for c_tags in tags:
        for c_tag in c_tags:
            tags_lst.append(c_tag)

    return tags_lst


def get_tag2idx_dict(tags):
    tag2idx = {}

    for i, tag in enumerate(tags):
        tag2idx[tag] = i

    return tag2idx


def display_worst10_confusion_matrix(pred_tags, true_tags):
    pred_tags_lst = extract_tags_list(pred_tags)
    true_tags_lst = extract_tags_list(true_tags)

    pred_unique_tags = list(set(pred_tags_lst))
    true_unique_tags = list(set(true_tags_lst))

    error_dict = {tag: 0 for tag in true_unique_tags}

    for pred_tag, true_tag in zip(pred_tags_lst, true_tags_lst):
        if pred_tag != true_tag:
            error_dict[true_tag] += 1

    tag_cnt_tups = list(error_dict.items())
    tag_cnt_tups.sort(key=lambda x: x[1])

    worst10_tags = [tup[0] for tup in tag_cnt_tups[-10:]]

    n, m = len(worst10_tags), len(pred_unique_tags)
    cm = np.zeros((n, m))

    worst10_tag2idx = get_tag2idx_dict(worst10_tags)
    pred_tag2idx = get_tag2idx_dict(pred_unique_tags)

    for pred_tag, true_tag in zip(pred_tags_lst, true_tags_lst):
        if true_tag in worst10_tags and pred_tag != true_tag:
            i = worst10_tag2idx[true_tag]
            j = pred_tag2idx[pred_tag]
            cm[i][j] += 1

    # remove columns with only zeros
    cols_to_remove, remained_pred_tags = [], []
    for j in range(m):
        col_j = cm[:, j]

        # if all the elements in column j are zeros, remove it
        if np.sum(col_j) == 0:
            cols_to_remove.append(j)

        # the pred tag remained in the confusion matrix
        else:
            remained_pred_tags.append(pred_unique_tags[j])

    # remove all the unwanted colmuns
    cm = np.delete(cm, cols_to_remove, axis=1)

    # display confusion matrix
    plt.figure(figsize=(15, 7))
    sns.heatmap(cm, cmap="Blues", annot=True, linewidth=0.5)

    plt.xticks(np.arange(0.5, cm.shape[1], 1), remained_pred_tags)
    plt.yticks(np.arange(0.5, cm.shape[0], 1), worst10_tags)

    plt.show()


def calc_accuracy(pred_tags, true_tags):
    pred_tags_lst = extract_tags_list(pred_tags)
    true_tags_lst = extract_tags_list(true_tags)

    correct_tags_num = sum([1 if pred_tag == true_tag else 0 for pred_tag, true_tag in zip(pred_tags_lst, true_tags_lst)])
    accuracy = (correct_tags_num / len(pred_tags_lst)) * 100

    return round(accuracy, 4)


def evaluate_predication_file(pred_file_path, true_file_path, acc_path, dataset_type):
    _, pred_tags = extract_sentences_and_tags(pred_file_path)
    _, true_tags = extract_sentences_and_tags(true_file_path)

    accuracy = calc_accuracy(pred_tags, true_tags)
    print_msg = f"Accuracy on the {dataset_type} set: {accuracy}%"
    print(print_msg)

    # display_worst10_confusion_matrix(pred_tags, true_tags)

    if acc_path is not None:
        res_output_file = open(acc_path, 'w')
        res_output_file.write(print_msg)
