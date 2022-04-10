import joblib
import logging
from main import get_corpus, featurize


if __name__ == '__main__':

    corpus = get_corpus(filepath="data/predict.txt")
    corpus_vec = featurize(corpus)

    model = joblib.load("model/oc_svm.model")

    logging.info("模型加载成功")

    predicts = model.predict(corpus_vec)
    print(predicts)
