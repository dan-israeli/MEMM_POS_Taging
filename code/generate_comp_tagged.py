from inference import tag_test_file
import pickle


def create_comp_file(trained_model_path, test_path, pred_path, B):

    # load the trained model
    with open(trained_model_path, 'rb') as f:
        optimal_weights, feature2id = pickle.load(f)

    # create a predications file for the test file
    tag_test_file(optimal_weights, feature2id, test_path, pred_path, B, tagged=False)


def pred_comp_file_model_1():
    trained_model_path = "comp_model_1_weights.pkl"
    test_path = "data/comp1.words"
    pred_path = "comp_m1.wtag"

    # chosen B for the inference
    B = 100

    create_comp_file(trained_model_path, test_path, pred_path, B)


def pred_comp_file_model_2():
    trained_model_path = "comp_model_2_weights.pkl"
    test_path = "data/comp2.words"
    pred_path = "comp_m2.wtag"

    # chosen B for the inference
    B = 100

    create_comp_file(trained_model_path, test_path, pred_path, B)


def main():
    # create the predication file for the first competition ('comp1.words')
    pred_comp_file_model_1()
    # create the predication file for the second competition ('comp2.words')
    pred_comp_file_model_2()


if __name__ == '__main__':
    main()
