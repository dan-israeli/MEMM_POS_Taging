from preprocessing import extract_sentences_and_tags, pad_sentences
from evaluation import evaluate_predication_file
from inference import tag_test_file, memm_viterbi
from optimization import get_optimal_vector
from preprocessing import preprocess_train
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
import pickle
import os


def combine_files(files_paths, output_path):

    with open(output_path, 'w') as f:
        for file_path in files_paths:
            with open(file_path) as g:
                f.write(g.read())

def train_model(train_path, save_path, threshold, lam):
    # extract the sentences (words) and the tags (POS) from the train file
    sentences, tags = extract_sentences_and_tags(train_path)

    # train the model
    statistics, feature2id = preprocess_train(sentences, tags, threshold)
    optimal_weights = get_optimal_vector(statistics, feature2id, lam)

    # save the trained model
    with open(save_path, 'wb+') as f:
        pickle.dump((optimal_weights, feature2id), f)


def evaluate_model(trained_model_path, test_path, pred_path, acc_path, B, dataset_type):
    # load the trained model
    with open(trained_model_path, 'rb') as f:
        optimal_weights, feature2id = pickle.load(f)

    # create a predications file for the test file
    tag_test_file(optimal_weights, feature2id, test_path, pred_path, B)

    # evaluate the predications
    evaluate_predication_file(pred_path, test_path, acc_path, dataset_type)


def cross_validation(sentences, tags, threshold, lam, B, cv_acc_path, fold_num,
                     main_sentences=None, main_tags=None):

    # convert to numpy array (used for cross-validation)
    sentences = np.array(sentences, dtype=object)
    tags = np.array(tags, dtype=object)

    # create cross validation object with 'fold_num' folds
    kf = KFold(n_splits=fold_num)

    accuracy_sum = 0
    # iterate over the different folds of the data
    for i, (train_folds_idx, test_fold_idx) in enumerate(kf.split(sentences), start=1):
        print(f"current fold: {i}")

        # get train-folds
        c_fold_train_sentences = list(sentences[train_folds_idx])
        c_fold_train_tags = list(tags[train_folds_idx])

        # add main_sentences and main_tags as the base train data of all train folds (in every iteration)
        if main_sentences is not None:
            c_fold_train_sentences += main_sentences
            c_fold_train_tags += main_tags

        # get test-fold
        c_fold_test_sentences = list(sentences[test_fold_idx])
        c_fold_test_tags = list(tags[test_fold_idx])

        # train the model on the train-folds
        c_fold_statistics, c_fold_feature2id = preprocess_train(c_fold_train_sentences, c_fold_train_tags, threshold)
        c_fold_optimal_weights = get_optimal_vector(c_fold_statistics, c_fold_feature2id, lam)

        c_fold_test_padded_sentences = pad_sentences(c_fold_test_sentences)
        c_fold__test_sentences_num = len(c_fold_test_sentences)

        c_fold_total_correct, c_fold_total_tags = 0, 0
        for c_sentence, c_tags in tqdm(zip(c_fold_test_padded_sentences, c_fold_test_tags), total=c_fold__test_sentences_num):
            c_pred_tags = memm_viterbi(tuple(c_sentence), c_fold_optimal_weights, c_fold_feature2id, B)

            c_fold_total_correct += sum(1 if pred_tag == true_tag else 0 for pred_tag, true_tag in zip(c_pred_tags, c_tags))
            c_fold_total_tags += len(c_tags)

        # calculate the accuracy of the current fold
        c_fold_accuracy = c_fold_total_correct / c_fold_total_tags
        accuracy_sum += c_fold_accuracy

    # calculate the average accuracy over all the test-folds
    avg_accuracy = round(((accuracy_sum / fold_num) * 100), 4)


    print_msg = f"Average accuracy: {avg_accuracy}"
    print(print_msg)

    # write the result into a file
    if cv_acc_path is not None:
        res_output_file = open(cv_acc_path, 'w')
        res_output_file.write(print_msg)


def train_comp_model_1():
    train_path = "data/train1_and_test1.wtag"
    save_path = "comp_model_1_weights.pkl"


    # combine train1 and test1 files into one combined train file
    files_paths = ["data/train1.wtag", "data/test1.wtag"]
    combine_files(files_paths, train_path)

    # chosen hyper-parameters
    threshold, lam = 1, 0.3

    train_model(train_path, save_path, threshold, lam)


def train_comp_model_2():
    train_path = "data/train2.wtag"
    save_path = "comp_model_2_weights.pkl"

    # chosen hyper-parameters
    threshold, lam = 1, 0.13

    train_model(train_path, save_path, threshold, lam)


def model_1():
    # chosen hyper-parameters
    threshold, lam = 1, 0.3

    train_path = "data/train1.wtag"
    save_path = "model_1_train1_weights.pkl"
    trained_model_path = "model_1_train1_weights.pkl"

    # train the model (only on train1) and save it
    if save_path is not None:
        train_model(train_path, save_path, threshold, lam)

    # evaluate model 1 on the train set
    test_path = "data/train1.wtag"
    pred_path = "model_1_train1_preds.wtag"
    acc_path = "model_1_train1_accuracy_results.txt"

    evaluate_model(trained_model_path, test_path, pred_path, acc_path, B=1, dataset_type="train")

    # evaluate model 1 on the test set
    test_path = "data/test1.wtag"
    pred_path = "model_1_test1_preds.wtag"
    acc_path = "model_1_test1_accuracy_results.txt"

    evaluate_model(trained_model_path, test_path, pred_path, acc_path, B=10, dataset_type="test")


def model_2():
    # chosen hyper-parameters
    threshold, lam = 1, 0.3

    train_path = "data/train2.wtag"
    save_path = "model_2_train2_weights.pkl"
    trained_model_path = "model_2_train2_weights.pkl"

    # train the model (on train2) and save it
    if save_path is not None:
        train_model(train_path, save_path, threshold, lam)

    # evaluate model 2 on the train set
    test_path = "data/train2.wtag"
    pred_path = "model_2_train2_preds.wtag"
    acc_path = "model_2_train2_accuracy_results.txt"

    evaluate_model(trained_model_path, test_path, pred_path, acc_path, B=10, dataset_type="train")


def model_1_cv():
    """
    This function is used to determine if training model 1 on more data (than 'train1') will increase its performance.

    2-fold cross validation on 'test1' data, and make 'train1' data as the base train data of all folds. In each iteration:
    - the train folds data contains 5500 sentences - 5000 from 'train1' and 500 from of the two folds of 'test1'
    - The test folds data contains 500 sentences - the second fold of 'test1'
    """
    # chosen hyper-parameters
    threshold, lam = 1, 0.3
    B = 10

    main_train_path = "data/train1.wtag"
    cv_train_path = "data/test1.wtag"
    cv_acc_path = "model_1_cv_accuracy_results.txt"

    main_sentences, main_tags = extract_sentences_and_tags(main_train_path)
    cv_sentences, cv_tags = extract_sentences_and_tags(cv_train_path)
    fold_num = 2

    # cross validation on test1 data, and train1 data is the base train data of all folds
    # in each iteration the train_folds data contains 5500 sentences and the
    cross_validation(cv_sentences, cv_tags, threshold, lam, B, cv_acc_path, fold_num,
                     main_sentences, main_tags)


def model_2_cv():

    # chosen hyper-parameters
    threshold, lam = 1, 0.13
    B = 10

    data_path = "data/train2.wtag"
    cv_acc_path = "model_2_cv_accuracy_results.txt"

    # extract the sentences (words) and the tags (POS) from the train file
    sentences, tags = extract_sentences_and_tags(data_path)

    # perform cross validation and calculate the average accuracy across all folds
    cross_validation(sentences, tags, threshold, lam, B, cv_acc_path, fold_num=5)


def main():
    # train_comp_model_1() # uncomment to train the model of competition 1
    # train_comp_model_2() # uncomment to train the model of competition 2

    # model_1() # uncomment to evaluate model's 1 performance on train1 and test1 (train1 is the only train data)
    # model_2() # uncomment to evaluate model's 2 performance on train2

    # model_1_cv() # uncomment to run "model_1_cv" (for information, see docstring)
    # model_2_cv() # uncomment to evaluate model's 2 performance using cross validation on train2

    pass


if __name__ == '__main__':
    main()

