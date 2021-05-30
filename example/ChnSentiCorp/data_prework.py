'''
ChnSentiCorp酒店评价数据集由谭松波从携程网上收集整理得到，根据用户评价内容将其分为正向与负向。阅读源数据后可知，正向评价的标准是较为宽松的，只有措辞十分严厉的才会被列为负向。
本代码将原始数据处理为train、test、valid三组以符合本项目的读取规范，不对数据做其他处理。
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('example/ChnSentiCorp/ChnSentiCorp_htl_all.csv')
train, test = train_test_split(data, test_size=0.3); valid, test = train_test_split(test, test_size=0.5)
print (train.shape, test.shape, valid.shape)

names = ['train','test','valid']
data_list = [train, test, valid]
for i in range(3):
    data_list[i].columns = ['label','text']
    data_list[i].to_csv('example/ChnSentiCorp/ChnSentiCorp_' + names[i] + '.csv', index=None)