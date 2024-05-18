import pandas as pd
import numpy as np
 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 数据处理
proccessed = True
def onehotmodel(data):
    num_col = data.select_dtypes(include=[np.number]) # 数值型数据的列数
    non_num_col = data.select_dtypes(exclude=[np.number]) # 非数值型数据的列数
    onehotnum = pd.get_dummies(non_num_col) # 对非数值型数据进行独热编码
    data = pd.concat([num_col, onehotnum], axis=1) # 拼接数据
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


abc = AdaBoostClassifier(random_state=42)  # adaboost
abcmodel = abc.fit(x_train, y_train)  # 拟合训练集
y_pred = abcmodel.predict(x_test)
 
gbdt = GradientBoostingClassifier(random_state=42)  # gbdt
gbdtmodel = gbdt.fit(x_train, y_train)  # 拟合训练集
y_pred = gbdtmodel.predict(x_test)
 
y_pred = abcmodel.predict(x_test)
print("ABC验证数据的准确率: ", accuracy_score(y_test, y_pred))
print("ABC验证数据的精确率 ", precision_score(y_test, y_pred))
print("ABC验证数据的召回率 ", recall_score(y_test, y_pred))
print("ABC验证数据的F1值 ", f1_score(y_test, y_pred))
test_pred = abcmodel.predict(test_data)
test_pred = pd.DataFrame(test_pred, columns=['bad_good'])
sub = pd.concat([test_CUST_ID, test_pred], axis=1)
#sub.to_csv('submission.csv')

y_pred = gbdtmodel.predict(x_test)
print("GBDT验证数据的准确率: ", accuracy_score(y_test, y_pred))
print("GBDT验证数据的精确率 ", precision_score(y_test, y_pred))
print("GBDT验证数据的召回率 ", recall_score(y_test, y_pred))
print("GBDT验证数据的F1值 ", f1_score(y_test, y_pred))
test_pred = gbdtmodel.predict(test_data)
test_pred = pd.DataFrame(test_pred, columns=['bad_good'])
sub = pd.concat([test_CUST_ID, test_pred], axis=1)
sub.to_csv('submission.csv')