# 深度学习课程竞赛实验报告

## 小组成员及分工

PB21111710 赵瑞冬：代码测试、调参，报告撰写

PB21111716 李乐禛：实验代码编写，代码debug

PB21111727 江岩：模型搜集，代码编写、调参

## 实验选题

本小组选取数据科学竞赛中的**赛题四：用户逾期行为预测**。

我们通过信贷用户的其他行为，预测用户是否逾期。

## 模型介绍

本小组共实现了`XGBoost`,`AdaBoost`,`GBDT`,`RFC`以及决策树模型的实现。

### 决策树

基尼系数Gini：衡量选择标准的不确定程度；
以基尼系数为核心的决策树称为CART决策树(Classification and Regression Tree)。以下所有的决策树都是CART决策树。其目标是创建一个模型，通过学习从数据特征中推断出的简单决策规则来预测目标变量的值。一棵树可以看作是一个片断常数近似值。决策过程的每一次判定都是对某一属性的“测试”，决策最终结论则对应最终的判定结果。一般一颗决策树包含：一个根节点、若干个内部节点和若干个叶子节点。步骤如下：

- 第一步：开始时将所有记录看作一个节点。
- 第二步：遍历每个变量的每一种分割方式，找到最好的分割点。
- 第三步：分割成两个节点N1和N2。
- 第四步：对N1和N2分别继续执行第二步和第三步，直到每个节点足够“纯”为止。

优点：非常直观，可解释极强； 预测速度非常快；既可以处理离散值也可以处理连续值，还可以处理缺失值。

缺点：容易过拟合；需要处理样本不均衡的问题；样本的变化会引发树结构巨变。

### XGBoost

XGBoost全称为*eXtreme Gradient Boosting*，即极致梯度提升树。
XGBoost是*Boosting*算法的其中一种，Boosting算法的思想是将许多弱分类器集成在一起，形成一个*强分类器*。

XGBoost的基本组成元素是决策树，这些决策树即为“弱学习器”，它们共同组成了XGBoost；
并且这些组成XGBoost的决策树之间是有先后顺序的：后一棵决策树的生成会考虑前一棵决策树的预测结果，即将前一棵决策树的偏差考虑在内，使得先前决策树做错的训练样本在后续受到更多的关注，然后基于调整后的样本分布来训练下一棵决策树。

优点：用到了二阶导数，精度高；灵活性强；且有效防止过拟合；可以并行计算

缺点：预排序过程的时间与空间复杂度都较高。

### GBDT

GBDT也属于Boosting算法的一种，通过多轮迭代,每轮迭代产生一个弱分类器，每个分类器在上一轮分类器的残差基础上进行训练。对弱分类器的要求一般是足够简单，并且是低方差和高偏差的。因为训练的过程是通过降低偏差来不断提高最终分类器的精度。

弱分类器一般会选择为CART TREE（也就是分类回归树）。由于上述高偏差和简单的要求，每个分类回归树的深度不会很深。最终的总分类器是将每轮训练得到的弱分类器加权求和得到的（也就是加法模型）。

优点：非线性变换比较多，表达能力强，而且不需要做复杂的特征工程和特征变换。

缺点：Boost是一个串行过程，不好并行化，而且计算复杂度高，同时不太适合高维稀疏特征；传统GBDT在优化时只用到一阶导数信息。

### AdaBoost

与上两种情况类似，也是Boosting算法的一种，同样也是先训练不同的弱分类器然后合并成一个强分类器，但是在各个弱分类器的训练过程结束后，加大分类误差率小的弱分类器的权重，使其在最终的分类函数中起着较大的决定作用，而降低分类误差率大的弱分类器的权重，使其在最终的分类函数中起着较小的决定作用。换言之，误差率低的弱分类器在最终分类器中占的权重较大，否则较小。

优点：准确性强，易于实现，自动处理特征选择，较灵活，不容易过拟合

缺点：对噪声和异常值敏感，计算量较大，在面对极端不平衡数据时表现不佳。

### RFC（随机森林分类器）

RFC算法，全称为Random Forest Classification，是一种基于决策树的集成学习算法。它通过构建多个决策树模型，并通过投票或取平均值的方式来进行分类预测。

   RFC算法通过构建多个决策树来形成随机森林。每个决策树都是独立训练的，它们之间没有关联。当需要进行分类预测时，每个决策树都会给出一个预测结果，最后通过投票或取平均值的方式来确定最终的预测结果。

优点：可以处理大量的输入；它可以对于一般化后的误差产生无偏估计；如果有很大一部分的资料遗失，仍可以维持准确度；学习过程快速

缺点：由于随机森林自身的随机性，导致预测结果波动 ; 样本基数大时，增加训练时间。

## 实验过程

- 首先，对数据进行预处理。
- 划分训练集与验证集。
- 分别调用不同的分类器进行处理，并进行提交结果和调参。
- 选用正确率最高的模型。

## 关键代码讲解

### 数据预处理

```python
proccessed = True

def onehotmodel(data):
    num_col = data.select_dtypes(include=[np.number])  # 数值型数据的列数
    non_num_col = data.select_dtypes(exclude=[np.number])  # 非数值型数据的列数
    onehotnum = pd.get_dummies(non_num_col)  # 对非数值型数据进行独热编码
    data = pd.concat([num_col, onehotnum], axis=1)  # 拼接数据
    return data

# 避免重复处理数据
if (proccessed):
    train_data = pd.read_csv('train_processed.csv')
    test_data = pd.read_csv('test_processed.csv')
else:
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    dropcols = [
        "OPEN_ORG_NUM", "IDF_TYP_CD", "GENDER", "CUST_EUP_ACCT_FLAG",
        "CUST_AU_ACCT_FLAG", "CUST_SALARY_FINANCIAL_FLAG",
        "CUST_SOCIAL_SECURITYIC_FLAG", "CUST_MTFLOW_FLAG", "CUST_DOLLER_FLAG",
        "CUST_INTERNATIONAL_GOLD_FLAG", "CUST_INTERNATIONAL_GOLD_FLAG",
        "CUST_INTERNATIONAL_SIL_FLAG", "CUST_INTERNATIONAL_DIAMOND_FLAG",
        "CUST_GOLD_COMMON_FLAG", "CUST_STAD_PLATINUM_FLAG",
        "CUST_LUXURY_PLATINUM_FLAG", "CUST_PLATINUM_FINANCIAL_FLAG",
        "CUST_DIAMOND_FLAG", "CUST_INFINIT_FLAG", "CUST_BUSINESS_FLAG"
    ]
    # 删除无用的行
    train_data.drop(dropcols, axis=1, inplace=True)
    test_data.drop(dropcols, axis=1, inplace=True)
    # 删除重复行
    train_data = train_data.drop_duplicates(keep="first")
    train_data.dropna(inplace=True)
    # 保存处理后的数据
    train_data.to_csv("train_processed.csv")
    test_data.to_csv("test_processed.csv")
# 提取训练集的bad_good
train_data_target = train_data['bad_good']
# 提取测试集的CUST_ID
test_CUST_ID = test_data['CUST_ID']
# 删除训练集中的bad_good列
train_data.drop(['bad_good'], axis=1, inplace=True)
# 独热编码
train_data = onehotmodel(train_data)
test_data = onehotmodel(test_data)
# 划分训练集、验证集
x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                    train_data_target,
                                                    test_size=0.1,
                                                    random_state=42)
```

预处理步骤如下：

1. **定义独热编码函数** (`onehotmodel`)：
   - 对非数值型数据使用`pd.get_dummies`进行独热编码，将分类变量转换为一系列二进制列，每一列代表一个类别。
2. **条件判断是否需要重新处理数据**：
   - 如果变量`proccessed`为`True`，则直接读取已经处理过的数据。
   - 如果`proccessed`为`False`，则从原始的`train.csv`和`test.csv`文件中读取数据，并进行以下处理：
     - 删除无关列。
     - 删除重复的行和含有缺失值的行。
3. **提取目标变量和特征**
4. **删除训练集中的目标变量列**
5. **对训练集和测试集应用独热编码**
6. **划分训练集和验证集**

### 模型构建与训练

主要采用`sklearn`包中的各个模块进行模型构建。如XGBoost：

```python
from xgboost import XGBClassifier
XGB = XGBClassifier(learning_rate=0.08,
                    n_estimators=50,
                    max_depth=5,
                    gamma=0,
                    subsample=0.9,
                    colsample_bytree=0.5)
model = XGB.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("XGB验证数据的准确率: ", accuracy_score(y_test, y_pred))
print("XGB验证数据的精确率 ", precision_score(y_test, y_pred))
print("XGB验证数据的召回率 ", recall_score(y_test, y_pred))
print("XGB验证数据的F1值 ", f1_score(y_test, y_pred))
test_pred = model.predict(test_data)
test_pred = pd.DataFrame(test_pred, columns=['bad_good'])
sub = pd.concat([test_CUST_ID, test_pred], axis=1)
sub.to_csv('XGB_submission.csv')
```

其余几个模型基本类似，不再赘述。

## 调参过程

针对XGBoost模型进行调参，结果如下：

```python
x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                    train_data_target,
                                                    test_size=0.2,
                                                    random_state=42)
XGB = XGBClassifier(learning_rate=0.08,
                    n_estimators=50,
                    max_depth=5,
                    gamma=0,
                    subsample=0.9,
                    colsample_bytree=0.5)
```

![image-20240513162359047](C:/Users/Administrator/Documents/Data-Foundation/figures/image-20240513162359047.png)

![image-20240513162433345](C:/Users/Administrator/Documents/Data-Foundation/figures/image-20240513162433345.png)

```python
x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                    train_data_target,
                                                    test_size=0.1,
                                                    random_state=42)
XGB = XGBClassifier(learning_rate=0.08,
                    n_estimators=50,
                    max_depth=5,
                    gamma=0,
                    subsample=0.9,
                    colsample_bytree=0.5)
```

![image-20240513161838348](C:/Users/Administrator/Documents/Data-Foundation/figures/image-20240513161838348.png)

![image-20240513161936673](C:/Users/Administrator/Documents/Data-Foundation/figures/image-20240513161936673.png)

```python
x_train, x_test, y_train, y_test = train_test_split(train_data,
                                                    train_data_target,
                                                    test_size=0.1,
                                                    random_state=42)
XGB = XGBClassifier(learning_rate=0.08,
                    n_estimators=50,
                    max_depth=3,
                    gamma=0,
                    subsample=0.9,
                    colsample_bytree=0.5)
```

![image-20240513163057835](C:/Users/Administrator/Documents/Data-Foundation/figures/image-20240513163057835.png)

![image-20240513163105998](C:/Users/Administrator/Documents/Data-Foundation/figures/image-20240513163105998.png)

## 实验结果

五种模型下，除了随机森林算法的F1值为0.997,其余四种模型在四舍五入后的F1值均为1.0，且提交后得分均接近1.0（还是就是1？？）由于决策树的模型较为简单，用时最短，其余用时较长。

可以看出本小组给出的五个模型都可以很好的完成题目的任务。

## 参考链接

1.XGBoost：https://blog.csdn.net/weixin_55073640/article/details/129519870

2.GBDT:https://blog.csdn.net/NXHYD/article/details/104601247

3.AdaBoost：https://blog.csdn.net/qq_34650787/article/details/83625916

4.随机森林：https://wenku.baidu.com/view/5238e97beb7101f69e3143323968011ca300f7ab.html

5.决策树：https://blog.csdn.net/ex_6450/article/details/126077545

