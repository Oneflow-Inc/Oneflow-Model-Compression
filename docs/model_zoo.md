# 模型库
# 1. 图像分类
## 1.1 量化
数据集：ImageNet

模型： Resnet-50

评价指标：准确率

<table>
<thead>
  <tr>
    <th>模型 <br>Resnet-50</th>
    <th>GPU个数/每个节点</th>
    <th>Batch Size/<br>每个节点</th>
    <th>Samples/s</th>
    <th>Top_1</th>
    <th>Top_5</th>
    <th>推理<br>加速</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Oneflow未量化</td>
    <td>1</td>
    <td>256</td>
    <td>482.22</td>
    <td>0.7732</td>
    <td>0.9357</td>
    <td>1.0x</td>
  </tr>
  <tr>
    <td>1</td>
    <td>350(max)</td>
    <td>483.12</td>
    <td>0.7732</td>
    <td>0.9357</td>
    <td>1.0x</td>
  </tr>
  <tr>
    <td>TensorRT<br>Online int 8<br>Calibration</td>
    <td>1</td>
    <td>256</td>
    <td>1357.99</td>
    <td>0.7731</td>
    <td>0.9356</td>
    <td>2.8x</td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT<br>Offline int 8<br>Calibration</td>
    <td>1</td>
    <td>256</td>
    <td>1319.04</td>
    <td>0.7721</td>
    <td>0.9347</td>
    <td>2.7x</td>
  </tr>
  <tr>
    <td>1</td>
    <td>350</td>
    <td>1443.31</td>
    <td>0.7722</td>
    <td>0.9348</td>
    <td>3.0x</td>
  </tr>
  <tr>
    <td rowspan="2">TensorRT<br>FP32</td>
    <td>1</td>
    <td>256</td>
    <td>780.61</td>
    <td>0.7731</td>
    <td>0.9356</td>
    <td>1.6x</td>
  </tr>
  <tr>
    <td>1</td>
    <td>350</td>
    <td>785.00</td>
    <td>0.7732</td>
    <td>0.9357</td>
    <td>1.6x</td>
  </tr>
</tbody>
</table>

## 1.2 剪枝

数据集：Cifar10

模型：Alexnet、Lenet

设置：剪枝率为0.5、0.7

|     模型 - 剪枝算子     | 测试次数 |  Acc   | 剪枝率 | 压缩比例 | 推理耗时samples/s |
| :---------------------: | :------: | :----: | :----: | :------: | :---------------: |
|    Alexnet - 无剪枝     |    5     | 94.89% |   -    |    1x    |       5409        |
|      Alexnet - bn       |    5     | 98.81% |  50%   |   1.4x   |       5968        |
|   Alexnet - conv_all    |    5     | 93.95% |  50%   |   1.3x   |       5969        |
|   Alexnet - conv_avg    |    5     | 98.56% |  50%   |   1.3x   |       5865        |
|   Alexnet - conv_max    |    5     | 97.44% |  50%   |   1.3x   |       5555        |
|    Alexnet - random     |    5     | 97.32% |  50%   |   1.3x   |       5580        |
| Alexnet -conv_threshold |    5     | 98.03% |  50%   |  x1.3x   |       5567        |
|     Lenet - 无剪枝      |    5     | 75.72% |   -    |    1x    |       5821        |
|       Lenet - bn        |    5     | 64.89% |  70%   |    3x    |       1923        |

# 2. 文本分类
## 2.1 知识蒸馏
数据集：SST-2

环境：单卡2080Ti

设置：BERT类模型最大序列长度设为128，LSTM类模型最大序列长度设为32，词表大小为10000

| 模型 | 测试次数 | Acc | 层数 | 隐藏层维度/前馈层维度 | 模型尺寸 | 压缩比例 | 推理耗时 | 推理加速 |
|:--:|:---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| BERT_base(Teacher) | 5 | 92.2% | 12 | 768/3072 | 110M | 1x | 4.04s | 1x |
| KD | 5 | 80.5% | 3 | 312/1200 | 14.5M | 7.5x | 0.81s | 5.0x |
| BiLSTM | 5 | 80.4% | 1 | 300/400 | 15.3M | 7.2x | 0.83s | 4.8x |
| Distilled-BiLSTM | 5 | 82.9% | 1 | 300/400 | 15.3M | 7.2x | 0.83s | 4.8x |
| BERT-PKD(from scratch) | 5 | 81.5% | 3 | 768/3072 | 45.7M | 2.4x | 1.69s | 2.4x |
| BERT-PKD | 5 | 88.4% | 3 | 768/3072 | 45.7M | 2.4x | 1.69s | 2.4x |
| TinyBERT | 5 | 91.3% | 4 | 312/1200 | 14.5M | 7.5x | 0.65s | 6.2x |
| BERT-of-Theseus | 5 | 87.2% | 4 | 768/3072 | 53.7M | 2.05x | 2.05s | 2.0x |

注：层数不包含embedding和prediction层。
