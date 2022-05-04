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

3.rest接口
```commandline
uvicorn rest:app --host 0.0.0.0 --port 8000
访问http://localhost:8000/SuicideDetector/xxxx
```


# 4.测试效果

### 模型识别样例
1.轻生倾向文本识别效果

label:
1表示轻生倾向，-1表示正常文本
```
1 手腕上有18道伤口，今晚过后又会多两道，我在等，等熬不下去的那一天逃离这个世界
1 街上家重度抑郁，我好想死真的好想好想，可是家人还在我死了他们怎么办。。每天就这样不停地折磨自己。。
1 FUMINQI没有人可以理解我现在的心情，处境。想不开那就算我是想不开吧，活着也没有什么意思。都怪我不够勇敢，赶紧来个意外结束掉吧。我快要崩溃啦。
1 如果不是今晚弟弟看我眼红红摸了我的头 我真的想死了算了 反正都在责怪我 都在觉的我做的不好 努力做有什么用
1 一想到自己迟早会死 我就莫名的心安
1 我生活的好难 我快坚持不下去了
1 对了大家好我是CC好累 想死
1 宠溺水三千小学的时候觉得死亡是件可怕的事情，但是不知怎么的初中到高中这六年了我想到的只有死亡
1 让人疲倦的梦我其实没有那么想死，只不过是不能活着了啊
1 温暖小葵花-这几天看了很多自杀方法，在放弃的自己的道路上越来越坚定。再也不想每天都活在无尽的崩溃里，生不如死
1 看不见尽头的日子 到底什么时候可以结束
-1 我的生命在倒计时
```

2.军事相关文本识别效果
```
1 咋又扯到军火
1 和你们军迷的一些要求还不一样
1 之后买不买拆腻子的中端货没有用了 不如重新去买毛子的低端货
1 还不如几个体格健壮点的 或者脑子好用的
1 那在大气层里飞的战斗机跃进到宇宙战斗机
1 没有不合适的伪装 只有不认真的准备工作
1 我国的情况和美国不一样
1 再加上北约协防
1 面具这玩意 没有什么特种不特种，只要不漏气，防护水平全看过滤罐
-1 那要能单机突出大气层，战斗机性能要求多高，还有要将推力降到多少驾驶员才不会被推死
-1 你怕是不了解虎斑
```

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