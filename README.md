# 中文文本分类项目
现有TextCNN，FastText，LSTM_Attention，DPCNN，基于Pytorch实现，缓慢更新中。

## 使用说明
在data文件夹下放入你需要分类的数据，命名格式为name_train.csv，name_valid.csv，name_test.csv，数据应有名为label, text的两列并用逗号隔开。
在parameter文件夹下的json文件中存储模型参数，可以自行调整。

训练模型：
```
python main.py -- DPCNN  # 可用模型：现有TextCNN，FastText，LSTM_Attention，DPCNN
```

训练完成的模型将保存于output文件夹。

## 数据说明

ChnSentiCorp店酒评价数据集，由谭松波从携程网上收集整理得到，根据用户评价内容将其分为正向与负向。
阅读源数据后可知，正向评价的标准是较为宽松的，只有措辞十分严厉的才会被列为负向。

## 模型效果

|  model   | accuracy  |
|  ----  | ----  |
| DPCNN  | 83.65% |
| TextCNN  | 84.65% |
| BiLSTM+Attention  | 84.32% |
| FastText  | 87.27% |

关于调参：
CNN的模型通道数对模型性能有较大的影响；词向量太长并不一定是好事；以及，dropout万用。
