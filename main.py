# 1.对文本进行预处理
# 2.使用laser对文本进行编码
# 3.使用oc训练模型
# 4.观察模型表现

# 其他
# 利用翻译测试，laser表征效果
# 利用NLI文本，测试laser表征效果
import os
import joblib
import requests
import logging
from matplotlib import pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def get_corpus(filepath):
    with open(filepath, "r", encoding="utf8") as fr:
        return [line.strip() for line in fr.readlines()]


def get_vect(query_in, lang='en', address='101.33.74.244:8050'):
    url = "http://" + address + "/vectorize"
    params = {"q": query_in, "lang": lang}
    resp = requests.get(url=url, params=params).json()
    return resp["embedding"]


def featurize(sentences):
    """基于laser的特征表示"""
    mul_line = "\n".join(sentences)
    print("laser is working")
    return get_vect(mul_line)


def suicide_corpus():
    train_pos = get_corpus(filepath="data/die.txt")  # 自杀倾向
    train_neg = get_corpus(filepath="data/normal.txt")           # 正常言论

    test_pos = get_corpus(filepath="data/die_test.txt")
    test_neg = get_corpus(filepath="data/normal_test.txt")
    return train_pos, train_neg, test_pos, test_neg


def get_feature(reuse=True):
    """
    获取向量化特征表示
    reuse: 是否利用已生成的文本向量特征，True：使用生成好的向量；False：重新生成新的向量特征
    """

    train_p_vec_path = "model/train_pos.vec"
    test_p_vec_path = "model/test_pos.vec"
    test_n_vec_path = "model/test_neg.vec"

    if reuse and (os.path.isfile(train_p_vec_path) and os.path.isfile(test_p_vec_path) and os.path.isfile(test_n_vec_path)):
        train_pos_vec = joblib.load(train_p_vec_path)
        test_pos_vec = joblib.load(test_p_vec_path)
        test_neg_vec = joblib.load(test_n_vec_path)
    else:
        train_pos, train_neg, test_pos, test_neg = suicide_corpus()
        # train_pos_vec = featurize(train_pos)
        train_pos_vec = featurize(train_pos)

        test_pos_vec = featurize(test_pos)
        test_neg_vec = featurize(test_neg)

        logging.info("特征向量保存...")
        joblib.dump(train_pos_vec, train_p_vec_path)
        joblib.dump(test_pos_vec, test_p_vec_path)
        joblib.dump(test_neg_vec, test_n_vec_path)

    return train_pos_vec, test_pos_vec, test_neg_vec


def train(train_vec, reuse=False, model_path="model/oc_svm.model"):
    """"""
    if reuse:
        model = joblib.load(model_path)
        logging.info("模型加载成功")
    else:

        model = OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
        model.fit(train_vec)

        if os.path.exists(model_path):
            os.remove(model_path)

        joblib.dump(model, model_path)
        logging.info("模型保存位置:", model_path)

    return model


def plot(predicts, golds):
    """模型识别效果可视化"""
    cm = confusion_matrix(golds, predicts, labels=[1, -1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, -1])

    disp.plot()
    plt.show()


if __name__ == '__main__':

    train_pos_vec, test_pos_vec, test_neg_vec = get_feature()

    model = train(train_vec=train_pos_vec)

    result1 = model.predict(test_pos_vec)
    result2 = model.predict(test_neg_vec)

    predicts = list(result1) + list(result2)
    golds = [1] * len(result1) + [-1] * len(result2)

    result = classification_report(golds, predicts)
    print(result)

    # plot(predicts=predicts, golds=golds)

