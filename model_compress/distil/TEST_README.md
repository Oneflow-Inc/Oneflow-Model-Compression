## 知识蒸馏
### 1. 结果汇总
环境：单卡2080Ti

设置：BERT类模型最大序列长度设为128，LSTM类模型最大序列长度设为32，词表大小为10000
<table>
<thead>
  <tr>
    <th colspan="3">Model</th>
    <th>SST-2</th>
    <th>QQP</th>
    <th>MRPC</th>
    <th>RTE</th>
    <th>CoLA</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>BERT_base (Reported)</td>
    <td></td>
    <td></td>
    <td>92.7%</td>
    <td>89.6%</td>
    <td>88.6%</td>
    <td>69.3%</td>
    <td>56.3%</td>
  </tr>
  <tr>
    <td rowspan="5">BERT_base (Teacher)</td>
    <td>Layers</td>
    <td>12</td>
    <td rowspan="5">92.0%</td>
    <td rowspan="5">91.0%</td>
    <td rowspan="5">86.8%</td>
    <td rowspan="5">69.4%</td>
    <td rowspan="5">58.7%</td>
  </tr>
  <tr>
    <td>Hidden Size</td>
    <td>768</td>
  </tr>
  <tr>
    <td>Feed-forward Size</td>
    <td>3072</td>
  </tr>
  <tr>
    <td>Model Size</td>
    <td>110M (1×)</td>
  </tr>
  <tr>
    <td>Inference Time</td>
    <td>4.04s (1×)</td>
  </tr>
  <tr>
    <td rowspan="5">KD</td>
    <td>Layers</td>
    <td>4</td>
    <td rowspan="5">80.8%</td>
    <td rowspan="5">80.6%</td>
    <td rowspan="5">68.3%</td>
    <td rowspan="5">54.2%</td>
    <td rowspan="5">12.0%</td>
  </tr>
  <tr>
    <td>Hidden Size</td>
    <td>312</td>
  </tr>
  <tr>
    <td>Feed-forward Size</td>
    <td>1200</td>
  </tr>
  <tr>
    <td>Model Size</td>
    <td>14.5M (7.5×)</td>
  </tr>
  <tr>
    <td>Inference Time</td>
    <td>0.81s (5.0×)</td>
  </tr>
  <tr>
    <td rowspan="5">Distilled-BiLSTM</td>
    <td>Layers</td>
    <td>1</td>
    <td rowspan="5">83.3%</td>
    <td rowspan="5">76.5%</td>
    <td rowspan="5">68.8%</td>
    <td rowspan="5">55.9%</td>
    <td rowspan="5">13.1%</td>
  </tr>
  <tr>
    <td>Hidden Size</td>
    <td>300</td>
  </tr>
  <tr>
    <td>Feed-forward Size</td>
    <td>400</td>
  </tr>
  <tr>
    <td>Model Size</td>
    <td>15.3M (7.2×)</td>
  </tr>
  <tr>
    <td>Inference Time</td>
    <td>0.83s (4.8×)</td>
  </tr>
  <tr>
    <td rowspan="5">BERT-PKD</td>
    <td>Layers</td>
    <td>3</td>
    <td rowspan="5">88.7%</td>
    <td rowspan="5">89.8%</td>
    <td rowspan="5">70.9%</td>
    <td rowspan="5">58.7%</td>
    <td rowspan="5">25.6%</td>
  </tr>
  <tr>
    <td>Hidden Size</td>
    <td>768</td>
  </tr>
  <tr>
    <td>Feed-forward Size</td>
    <td>3072</td>
  </tr>
  <tr>
    <td>Model Size</td>
    <td>45.7M (2.4×)</td>
  </tr>
  <tr>
    <td>Inference Time</td>
    <td>1.69s (2.4×)</td>
  </tr>
  <tr>
    <td rowspan="5">TinyBERT</td>
    <td>Layers</td>
    <td>4</td>
    <td rowspan="5">91.6%</td>
    <td rowspan="5">89.6%</td>
    <td rowspan="5">86.5%</td>
    <td rowspan="5">65.6%</td>
    <td rowspan="5">22.3%</td>
  </tr>
  <tr>
    <td>Hidden Size</td>
    <td>312</td>
  </tr>
  <tr>
    <td>Feed-forward Size</td>
    <td>1200</td>
  </tr>
  <tr>
    <td>Model Size</td>
    <td>14.5M (7.5×)</td>
  </tr>
  <tr>
    <td>Inference Time</td>
    <td>0.65s (6.2×)</td>
  </tr>
  <tr>
    <td rowspan="5">BERT-of-Theseus</td>
    <td>Layers</td>
    <td>3</td>
    <td rowspan="5">87.2%</td>
    <td rowspan="5">88.3%</td>
    <td rowspan="5">71.8%</td>
    <td rowspan="5">55.7%</td>
    <td rowspan="5"></td>
  </tr>
  <tr>
    <td>Hidden Size</td>
    <td>768</td>
  </tr>
  <tr>
    <td>Feed-forward Size</td>
    <td>3072</td>
  </tr>
  <tr>
    <td>Model Size</td>
    <td>53.7M (2.05×)</td>
  </tr>
  <tr>
    <td>Inference Time</td>
    <td>2.05s (2.0×)</td>
  </tr>
</tbody>
</table>

注：层数不包含embedding和prediction层。


### 2. 测试说明
测试脚本中变量含义：
- DATA_ROOT: GLUE数据集总路径
- dataset: 数据集名 | SST-2, CoLA, MRPC, QQP, RTE
- STUDENT_DIR: 学生模型保存路径
- RESULT_DIR: 测试结果json文件保存路径 （如果RESULT_DIR=""，则默认保存到模型保存路径下，results_eval.json）

训练好的学生模型断点可以在此处下载，并放到相关目录下（`./models/student_model/`）：
- 下载链接 https://pan.baidu.com/s/1TZBCMO5xSnFxbPt41qpf7A  密码: 5rd7
- BERT-theseus 下载链接 https://pan.baidu.com/s/1PkA3Y7s1ih8HcIaYpRQqNg  密码: haku

数据集在此下载，并放置到相关目录(`data/glue_data`)下：
- GLUE任务数据集链接: https://pan.baidu.com/s/1Im0uQM_V_zXkmQuHsFae0A 提取码: u64u

测试命令说明如下：
```
KD（Knowledge Distillation): 
    bash run_eval_student_kd.sh {哪个GPU} {数据集名称} {模型ID}
Distilled-BiLSTM:
    bash run_eval_student_distilled_lstm.sh {哪个GPU} {数据集名称} {模型ID}
BERT-PKD:
    bash run_eval_student_bert_pkd.sh {哪个GPU} {数据集名称} {模型ID}
TinyBERT:
    bash run_eval_student_tinybert.sh {哪个GPU} {数据集名称} {模型ID}
BERT-of-Theseus
    bash run_eval_theseus.sh {哪个GPU} {数据集名称} {模型ID}
```

评价指标：
- Accuracy（Acc）: SST-2, MRPC, QQP, RTE
- MCC (Matthews correlation coefficient): CoLA

### 3. 执行测试
需要将将训练好的断点放在模型保存路径下（默认为`./models/`），执行以下脚本进行测试：
- (1) 测试教师模型：
    - BERT-base
        - SST-2 (Acc: 92.0%)
        ```
        bash run_eval_teacher.sh 0 SST-2 SST-2_epoch-3_lr-2e-5_wd-0.0001
        ```
        - QQP (Acc: 91.0%)
        ```
        bash run_eval_teacher.sh 0 QQP QQP_epoch-5_lr-2e-5_wd-0.0001
        ```
        - MRPC (Acc: 86.8%)
        ```
        bash run_eval_teacher.sh 0 MRPC MRPC_epoch-5_lr-1e-5_wd-0.001
        ```
        - RTE (Acc: 69.4%)
        ```
        bash run_eval_teacher.sh 0 RTE RTE_epoch-5_lr-3e-5_wd-0.0001
        ```
        - CoLA (MCC: 58.7%)
        ```
        bash run_eval_teacher.sh 0 CoLA CoLA_epoch-5_lr-1e-5_wd-0.01
        ```   

- (2) 测试不同知识蒸馏模型：
    - KD（Knowledge Distillation）
        - SST-2 (Acc: 80.8%)
        ```
        bash run_eval_student_kd.sh 0 SST-2 bert-pkd_3_epoch-4_lr-2e-5_wd-0.0001_kd_alpha-0.2_kd_beta-10
        ```
        - QQP (Acc: 80.6%)
        ```
        bash run_eval_student_kd.sh 0 QQP bert-kd-distl_epoch-5_lr-5e-5_wd-0.0001_kd_alpha-0.8
        ```
        - MRPC (Acc: 68.3%)
        ```
        bash run_eval_student_kd.sh 0 MRPC bert-kd-distl_epoch-5_lr-2e-5_wd-0.001_kd_alpha-0.8
        ```
        - RTE (Acc: 54.2%)
        ```
        bash run_eval_student_kd.sh 0 RTE bert-kd-distl_epoch-5_lr-2e-5_wd-0.0001_kd_alpha-0.8
        ```
        - CoLA (MCC: 12.0%)
        ```
        bash run_eval_student_kd.sh 0 CoLA bert-kd-distl_epoch-70_lr-5e-5_wd-0.0001_kd_alpha-0.8
        ```       
    - Distilled-BiLSTM
        - SST-2 (Acc: 83.3%)
        ```
        bash run_eval_student_distilled_lstm.sh 0 SST-2 bert-lstm_32-distl_epoch-5_lr-1e-4_wd-0.0001_kd_alpha-0.7
        ```
        - QQP (Acc: 76.5%)
        ```
        bash run_eval_student_distilled_lstm.sh 0 QQP bert-distilled_lstm_epoch-10_lr-7e-5_wd-0.0001_kd_alpha-0.7
        ```
        - MRPC (Acc: 68.8%)
        ```
        bash run_eval_student_distilled_lstm.sh 0 MRPC bert-distilled_lstm_epoch-30_lr-5e-6_wd-0.001_kd_alpha-0.7
        ```
        - RTE (Acc: 55.9%)
        ```
        bash run_eval_student_distilled_lstm.sh 0 RTE bert-distilled_lstm_epoch-30_lr-5e-5_wd-0.0001_kd_alpha-0.7
        ```
        - CoLA (MCC: 13.1%)
        ```
        bash run_eval_student_distilled_lstm.sh 0 CoLA bert-distilled_lstm_epoch-100_lr-5e-5_wd-0.0001_kd_alpha-0.7
        ```       
    - BERT-PKD
        - SST-2 (Acc: 88.7%)
        ```
        bash run_eval_student_bert_pkd.sh 0 SST-2 bert-pkd_3_epoch-4_lr-2e-5_wd-0.0001_kd_alpha-0.2_kd_beta-10
        ```
        - QQP (Acc: 89.8%)
        ```
        bash run_eval_student_bert_pkd.sh 0 QQP bert-pkd_3_epoch-5_lr-5e-5_wd-0.0001_kd_alpha-0.2_kd_beta-10
        ```
        - MRPC (Acc: 70.9%)
        ```
        bash run_eval_student_bert_pkd.sh 0 MRPC bert-pkd_3_epoch-5_lr-2e-5_wd-0.001_kd_alpha-0.2_kd_beta-10
        ```
        - RTE (Acc: 58.7%)
        ```
        bash run_eval_student_bert_pkd.sh 0 RTE bert-pkd_3_epoch-5_lr-3e-5_wd-0.0001_kd_alpha-0.2_kd_beta-10
        ```
        - CoLA (MCC: 25.6%)
        ```
        bash run_eval_student_bert_pkd.sh 0 CoLA bert-pkd_3_epoch-100_lr-5e-5_wd-0.0001_kd_alpha-0.2_kd_beta-10
        ```       

    - TinyBERT
        - SST-2 (Acc: 91.6%)
        ```
        bash run_eval_student_tinybert.sh 0 SST-2 tinybert_epoch-4_lr-2e-5_wd-0.0001
        ```
        - QQP (Acc: 89.6)
        ```
        bash run_eval_student_tinybert.sh 0 QQP tinybert_epoch-5_lr-1e-4_wd-0.0001
        ```
        - MRPC (Acc: 86.5%)
        ```
        bash run_eval_student_tinybert.sh 0 MRPC tinybert_epoch-30_lr-2e-5_wd-0.001
        ```
        - RTE (Acc: 65.6%)
        ```
        bash run_eval_student_tinybert.sh 0 RTE tinybert_epoch-5_lr-2e-5_wd-0.0001
        ```
        - CoLA (MCC: 22.3%)
        ```
        bash run_eval_student_tinybert.sh 0 CoLA tinybert_epoch-100_lr-7e-5_wd-0.0001
        ```       
      

    - BERT-theseus
        
        You need to switch to the branch of *tianshu*.
        
        You need to change the *DATA_ROOT* and *model_load_dir* to your data directory and your model saved directory.

        - SST-2 (Acc: 87.2%)
        ```
        bash run_eval_theseus.sh SST-2
        ```
        - QQP (Acc: 88.3%)
        ```
        bash run_eval_theseus.sh QQP 
        ```
        - MRPC (Acc: 71.8%)
        ```
        bash run_eval_theseus.sh MRPC 
        ```
        - RTE (Acc: 55.7%)
        ```
        bash run_eval_theseus.sh RTE 
        ``` 
