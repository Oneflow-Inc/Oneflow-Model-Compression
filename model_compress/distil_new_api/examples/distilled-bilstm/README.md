# Distilled-BiLSTM
["Distilling task-specific knowledge from bert into simple neural networks"](https://arxiv.org/abs/1903.12136)论文的实现

Distilled BiLSTM的教师模型采用微调过的BERT，学生模型采用简单神经网络LSTM。
蒸馏的目标是KD loss，即仅使用软标签进行蒸馏，将BERT中的知识蒸馏到LSTM中。

## 1 依赖
- Python 3.7
- oneflow-cu101 0.5.0
- numpy 1.19.4

完整的环境可以通过以下命令安装：
  ```bash
conda create -n distil python=3.7
  ```

  ```
python3 -m pip install oneflow -f https://staging.oneflow.info/branch/master/cu101
  ```
> 注：以下操作时，根目录为`model_compress/distil_new_api`   

## 2. 数据获取
如何获取数据请查阅[这里](../../README.md#22-数据获取)

## 3. 微调教师模型
如何微调教师模型请查阅[这里](../../README.md#23-微调教师模型)
  
## 4. 蒸馏到学生模型
### 4.1 训练
执行以下脚本将教师模型蒸馏到学生模型：
- DATA_ROOT: GLUE数据集总路径
- dataset: 任务名
- FT_BERT_BASE_DIR: 在特定任务上微调过的教师模型路径
- STUDENT_DIR: 学生模型保存路径
- RESULT_DIR: 测试结果json文件保存路径 （如果为RESULT_DIR=""，则默认保存到模型保存路径下，results_eval.json）
- SERVE_FOR_ONLINE: 模型是否用于上线 （默认SERVE_FOR_ONLINE='False'，如果SERVE_FOR_ONLINE='True'，则删除清理模型保存路径中的无关变量，如教师模型参数和优化器参数等等）

> 最大序列长度为32，词表大小为10000

```bash
bash run_train_student_distilled_lstm.sh
```

### 4.2 测试

执行以下脚本进行测试：
- DATA_ROOT: GLUE数据集总路径
- dataset: 任务名
- STUDENT_DIR: 学生模型保存路径
- RESULT_DIR: 测试结果json文件保存路径 （如果为RESULT_DIR=""，则默认保存到模型保存路径下，results_eval.json）

```bash
bash run_eval_student_distilled_lstm.sh
```
