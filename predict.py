import os

import joblib
import logging
from main import get_corpus, featurize

root_dir = os.path.abspath(os.path.dirname(__file__))
label_dict = {"1": "军事相关", "-1": "正常言论"}


def predict_from_file(filepath):
    corpus = get_corpus(filepath)
    corpus_vec = featurize(corpus)

    model = joblib.load(os.path.join(root_dir, "model/military/oc_svm.model"))
    logging.info("模型加载成功")

    predicts = model.predict(corpus_vec)

    return [(label_dict.get(str(label)), text) for label, text in zip(predicts, corpus)]


def predict_one(sentence):

    sentence_vec = featurize([sentence])
    model = joblib.load(os.path.join(root_dir, "model/military/oc_svm.model"))

    logging.info("模型加载成功")
    predicts = model.predict(sentence_vec)

    return label_dict.get(str(predicts[0]))


if __name__ == '__main__':

    results = predict_from_file(filepath="data/predict.txt")
    # result = predict_one("我国的情况和美国不一样")

    for res in results:
        print(res)