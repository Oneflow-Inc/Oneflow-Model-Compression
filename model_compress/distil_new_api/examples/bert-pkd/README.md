# BERT-PKD
["Patient knowledge distillation for bert model compression"](https://arxiv.org/abs/1908.09355)的论文实现。

传统的KD会导致学生模型在学习的时候只是学到了教师模型最终预测的概率分布，而完全忽略了中间隐藏层的表示，从而导致学生模型过拟合，泛化能力不足。
BERT-PKD除了进行软标签蒸馏外，还对教师模型的中间层进行蒸馏。

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
### 4.1 教师模型中间层保存与转换
为了初始化一个更好的学生模型，我们可以利用教师模型的中间层参数来初始化学生模型，而不是随机初始化一个学生模型。

执行以下命令将教师模型的某些中间层参数提取并保存，用于初始化学生模型：
- FT_BERT_BASE_DIR: 在特定任务上微调过的教师模型路径
- TMP_STUDENT_DIR: 临时学生模型路径
- LAYER_LIST: 保存的层数，如"2,6,10"是保存教师模型的第2，6，10层，用来初始化学生模型的第1，2，3层参数
```bash
FT_BERT_BASE_DIR="./models/finetuned_teacher/SST-2_epoch-3_lr-2e-5_wd-0.0001/snapshot_best"
#FT_BERT_BASE_DIR="./models/finetuned_teacher/RTE_epoch-5_lr-3e-5_wd-0.0001/snapshot_best"
#FT_BERT_BASE_DIR="./models/finetuned_teacher/MRPC_epoch-5_lr-1e-5_wd-0.001/snapshot_best"
#FT_BERT_BASE_DIR="./models/finetuned_teacher/CoLA_epoch-5_lr-1e-5_wd-0.01/snapshot_best"
#FT_BERT_BASE_DIR="./models/finetuned_teacher/QQP_epoch-5_lr-2e-5_wd-0.0001/snapshot_best"

TMP_STUDENT_DIR='./models/student_model/bert_pkd_3/SST-2'
LAYER_LIST="2,6,10"
python3 examples/bert-pkd/bert-pkd_generate_student_model.py \
  --teacher_model=${FT_BERT_BASE_DIR} \
  --student_model=${TMP_STUDENT_DIR} \
  --layer_list=${LAYER_LIST}
```


### 4.2 训练
执行以下脚本将教师模型蒸馏到学生模型：
- DATA_ROOT: GLUE数据集总路径
- dataset: 任务名
- FT_BERT_BASE_DIR: 在特定任务上微调过的教师模型路径
- TMP_STUDENT_DIR: 临时学生模型路径（从教师模型中间层初始化时需要指定）
- STUDENT_DIR: 学生模型保存路径
- RESULT_DIR: 测试结果json文件保存路径 （如果为RESULT_DIR=""，则默认保存到模型保存路径下，results_eval.json）
- SERVE_FOR_ONLINE: 模型是否用于上线 （默认SERVE_FOR_ONLINE='False'，如果SERVE_FOR_ONLINE='True'，则删除清理模型保存路径中的无关变量，如教师模型参数和优化器参数等等）

```bash
bash run_train_student_bert_pkd.sh
```

### 4.3 测试
执行以下脚本进行测试：
- DATA_ROOT: GLUE数据集总路径
- dataset: 任务名
- STUDENT_DIR: 学生模型保存路径
- RESULT_DIR: 测试结果json文件保存路径 （如果为RESULT_DIR=""，则默认保存到模型保存路径下，results_eval.json）

```bash
bash run_eval_student_bert_pkd.sh
```


