import os

import joblib
import logging
from main import get_corpus, featurize

root_dir = os.path.abspath(os.path.dirname(__file__))
label_dict = {"1": "轻生倾向", "-1": "正常言论"}


def predict_from_file(filepath):
    corpus = get_corpus(filepath)
    corpus_vec = featurize(corpus)

    model = joblib.load(os.path.join(root_dir, "model/oc_svm.model"))
    logging.info("模型加载成功")

    predicts = model.predict(corpus_vec)

    return [(label_dict.get(str(label)), text) for label, text in zip(predicts, corpus)]


def predict_one(sentence):

    sentence_vec = featurize([sentence])
    model = joblib.load(os.path.join(root_dir, "model/oc_svm.model"))

    logging.info("模型加载成功")
    predicts = model.predict(sentence_vec)

    return label_dict.get(str(predicts[0]))


if __name__ == '__main__':

    # results = predict_from_file(filepath="data/predict.txt")
    result = predict_one("这个世界已经不值得留恋，我只想赶紧死去")
    print(result)
