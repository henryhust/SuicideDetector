## 面向多语种的自杀倾向分类器

采用联合多语言句子表示的架构（LASER），将所有语言一同嵌入到一个独立的共享空间中（而不是为每一种语言都创建一个单独的模型），从而实现在 90 多种语言中的应用。

数据中短文本包含两类，即要么是正常微博、要么是自杀倾向微博。分类器采用OneClassSVM——单分类器，只需要具有自杀倾向的微博样本参与模型训练，随后可以在正常微博内容和具有自杀倾向的微博内容上进行测试。

数据集来源：https://github.com/Ckend/suicide-detect-svm

# 1.数据准备
数据集整体上分两个部分，一部分是训练集、一部分是测试集。其中，训练集和测试集中还要分为正常微博短文本和自杀倾向短文本。

自杀倾向的训练文本300条，正常文本10000条。测试集中自杀倾向微博和普通微博各50条。

每条微博按行存储在txt文件里。训练集中，正常微博命名为normal.txt, 自杀倾向微博命名为die.txt。测试集存放在后缀为_test.txt的文件中：

# 2.环境准备
2.1 python环境
```
    pip install -r requirements.txt
```

2.2 多语言向量表征 layser
```
拉取镜像
docker pull  19981002/laser-server
运行镜像
docker run -p 8050:80 -it 19981002/laser-server python app.py
```

# 3.训练
特征表示：采用LASER框架生成文本向量，采用http方式进行请求。
```
main.py
line23
def get_vect(query_in, lang='en', address='xxx.xxx.xxx.xxx:8080'):  <-此处address修改为自己的ip+端口号
    url = "http://" + address + "/vectorize"
    params = {"q": query_in, "lang": lang}
    resp = requests.get(url=url, params=params).json()
    return resp["embedding"]
```

模型训练：调用sklearn工具包中OneClassSVM分类器，使用自杀倾向训练文本进行训练。

```
    model = OneClassSVM(nu=0.1, kernel="rbf", gamma='auto')
    model.fit(train_vec)
```
模型保存路径：model/oc_svm.model

OneClassSvm API可参考：https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html?highlight=oneclasssvm#sklearn.svm.OneClassSVM

程序运行
```
1.模型训练
python main.py

2.模型预测
python predict.py
```

# 4.测试效果

### 中文语料

模型分别对两个类别的测试样本进行预测，计算分类精确率和召回率。scikit-learn提供了一个非常好用的函数classification_report来计算它们：

 
结果如下,1表示自杀倾向数据，-1表示正常文本数据：
```
              precision    recall  f1-score   support

          -1       0.86      0.84      0.85        50
           1       0.84      0.86      0.85        50

    accuracy                           0.85       100
   macro avg       0.85      0.85      0.85       100
weighted avg       0.85      0.85      0.85       100
```

分类器在自杀倾向样本和正常样本上均取得85%的F1值。


### 英文语料

将具有自杀倾向的中文文本翻译成英文，使用模型进行预测，发现模型效果十分差，模型无法识别出任何一条英文数据为自杀倾向文本（正确率为0），模型优化仍需继续探究。