## 面向多语种的自杀倾向分类器

一个自杀倾向分类器，这样的分类器如果应用得当，将可以帮助成千上万误入歧途的人们挽回生命。

为了简化问题，我们将短文本分为两种类别中的一种，即要么是正常微博、要么是自杀倾向微博。这样，有了上次的微博树洞，训练集和测试集就非常好获得了。由于是短文本二分类问题，可以使用scikit-learn的SVM分类模型。

不过要注意的是，我们的分类器并不能保证分类出来的结果百分百正确，毕竟心理状态是很难通过文本准确识别出来的，我们只能通过文字，大致判断其抑郁情况并加以介入。实际上这是一个宁可错杀一百，不可放过一个的问题。毕竟放过一个，可能就有一条生命悄然流逝。


# 1.数据准备
数据集整体上分两个部分，一部分是训练集、一部分是测试集。其中，训练集和测试集中还要分为正常微博短文本和自杀倾向短文本。

将上一篇爬取微博树洞的文章中得到的数据进行人工筛选后，挑出300条作为训练集（有点少，其实业界至少也要3000条以上），再根据上次的微博爬虫随意爬取10000条微博作为训练集的正常微博类。另外再分别搜集自杀倾向微博和普通微博各50条作为测试集。

每条微博按行存储在txt文件里。训练集中，正常微博命名为normal.txt, 自杀倾向微博命名为die.txt。测试集存放在后缀为_test.txt的文件中：

![Python 短文本自动识别个体是否有自杀倾向](https://pythondict.com/wp-content/uploads/2019/11/2019111013222864.png)

此外，接下来我们会使用到一个机器学习工具包叫scikit-learn(sklearn)，其打包好了许多机器学习模型和预处理的方法，方便我们构建分类器，在CMD/Terminal输入以下命令安装：

`pip install -U scikit-learn`

如果你还没有安装Python，请看[这篇文章安装Python](https://pythondict.com/how-to-install-python/)，然后再执行上述命令安装sklearn.

# 2.训练
使用scikit-learn的SVM分类模型，我们能很快滴训练并构建出一个分类器：

```
print('(3) SVM...')
from sklearn.svm import SVC
 
# 使用线性核函数的SVM分类器，并启用概率估计（分别显示分到两个类别的概率如：[0.12983359 0.87016641]）
svclf = SVC(kernel = 'linear', probability=True) 
 
# 开始训练
svclf.fit(x_train,y_train)
# 保存模型
joblib.dump(svclf, "model/die_svm_20191110.m")
```
这里我们忽略了SVM原理的讲述，SVM的原理可以参考这篇文章：[支持向量机（SVM）——原理篇](https://zhuanlan.zhihu.com/p/31886934)

# 4.测试
测试的时候，我们要分别计算模型对两个类别的分类精确率和召回率。scikit-learn提供了一个非常好用的函数classification_report来计算它们：

```
# 测试集进行测试
preds = svclf.predict(x_test)
y_preds = svclf.predict_proba(x_test)
 
preds = preds.tolist()
for i,pred in enumerate(preds):
    # 显示被分错的微博
    if int(pred) != int(y_test[i]):
        try:
            print(origin_eval_text[i], ':', test_texts[i], pred, y_test[i], y_preds[i])
        except Exception as e:
            print(e)
 
# 分别查看两个类别的准确率、召回率和F1值
print(classification_report(y_test, preds)) 
```

4.结果：

![Python 短文本自动识别个体是否有自杀倾向](https://pythondict.com/wp-content/uploads/2019/11/2019111013521068.png)

对自杀倾向微博的分类精确率为100%，但是查全率不够，它只找到了50条里的60%，也就是30条自杀倾向微博。

对于正常微博的分类，其精确率为71%，也就是说有部分正常微博被分类为自杀倾向微博，不过其查全率为100%，也就是不存在不被分类的正常微博。

这是建立在训练集还不够多的情况下的结果。我们的自杀倾向微博的数据仅仅才300条，这是远远不够的，如果能增加到3000条，相信结果会改进不少，尤其是对于自杀倾向微博的查全率有很大的帮助。预估最终该模型的精确率和召回率至少能达到95%。

本文源代码： https://github.com/Ckend/suicide-detect-svm 欢迎一同改进这个项目。如果你访问不了github，请关注文章最下方公众号，回复自杀倾向检测获得本项目完整源代码。

如果你喜欢今天的Python 教程，请持续关注[Python实用宝典](https://pythondict.com)，如果对你有帮助，麻烦在下面点一个赞/在看哦有任何问题都可以在下方留言区留言，我们会耐心解答的！

![Python 短文本自动识别个体是否有自杀倾向](https://pythondict.com/wp-content/uploads/2019/08/2019080218203145.jpg)
