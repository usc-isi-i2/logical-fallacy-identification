# import argparse


# def get_arguments():
#     parser = argparse.ArgumentParser(description="Text Classifier Trainer")
#     parser.add_argument(
#         "-m", help="type of model to be trained: naivebayes, perceptron", type=str)
#     parser.add_argument(
#         "-i", help="path of the input file where training file is in the form <text>TAB<label>", type=str)
#     parser.add_argument(
#         "--dev", help="path of the input file where evaluation file is in the form <text>", type=str
#     )
#     parser.add_argument(
#         "--devlabels", help="path of the input file where evaluation true labels file is in the form <label>", type=str
#     )
#     parser.add_argument(
#         "--epochs", help='Number of epochs for the training stage', type=int, default=80
#     )
#     parser.add_argument(
#         "--ngram", help="cap of the ngram getting used for the bag of words featurization", type=int, default=3
#     )
#     parser.add_argument(
#         '--features', help="Feature used for training", default="bow", type=str
#     )

#     parser.add_argument(
#         '--wandb', help="Wandb name when logging", default="normal", type=str
#     )

#     parser.add_argument(
#         '--decrypt', help="whether to decrypt the content of the dataset or not",
#         action=argparse.BooleanOptionalAction
#     )
#     parser.add_argument("-o", help="path of the file where the model is saved")

#     return parser.parse_args()


# if __name__ == "__main__":
#     args = get_arguments()
