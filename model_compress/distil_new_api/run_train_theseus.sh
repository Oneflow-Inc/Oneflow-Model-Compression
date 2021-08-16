# Copyright (c) The OneFlow Authors.
# Licensed under the Apache License

# Script for BERT-of-Theseus.

# ofrecord dataset dir
DATA_ROOT=../data/glue_ofrecord_test

dataset=SST-2

STUDENT_NAME=theseus

# saved student model dir
STUDENT_DIR="./models/student_model/${dataset}/${STUDENT_NAME}"
# tran log out
TRAIN_LOG_DIR=./log

# inference json result out
RESULT_DIR=$STUDENT_DIR

# fine-tuned teacher model dir
FT_BERT_BASE_DIR="/usr/local/output/model/before/snapshot_best"

# temp student model dir
TMP_STUDENT_DIR="./models/bert_theseus/${dataset}"

train_data_dir=$DATA_ROOT/${dataset}/train
eval_data_dir=$DATA_ROOT/${dataset}/eval

REPLACING_RATE=0.8
STUDENT_NUM_HIDDEN_LAYERS=3

# which GPU to use
GPU=2

if [ $dataset = "CoLA" ]; then
  train_example_num=8551
  eval_example_num=1043
  test_example_num=1063
  learning_rate=5e-5
  wd=0.0001
  epoch=70
elif [ $dataset = "MRPC" ]; then
  train_example_num=3668
  eval_example_num=408
  test_example_num=1725
  learning_rate=2e-5
  epoch=5
  wd=0.001
elif [ $dataset = "SST-2" ]; then
  train_example_num=67349
  eval_example_num=872
  test_example_num=1821
  learning_rate=2e-5
  epoch=4
  wd=0.0001
elif [ $dataset = "QQP" ]; then
  train_example_num=363849
  eval_example_num=40430
  test_example_num=0
  learning_rate=5e-5
  epoch=5
  wd=0.0001
elif [ $dataset = "MNLI" ]; then
  train_example_num=392702
  eval_example_num=9815
  test_example_num=0
  learning_rate=2e-5
  epoch=5
  wd=0.0001
elif [ $dataset = "WNLI" ]; then
  train_example_num=635
  eval_example_num=71
  test_example_num=0
  learning_rate=2e-5
  epoch=5
  wd=0.0001
elif [ $dataset = "RTE" ]; then
  train_example_num=2490
  eval_example_num=277
  test_example_num=0
  learning_rate=2e-5
  epoch=5
  wd=0.0001
else
  echo "dataset must be GLUE such as 'CoLA','MRPC','SST-2','QQP','MNLI','WNLI','STS-B',"
  exit
fi

CUDA_VISIBLE_DEVICES=$GPU python3 ./examples/BERT-of-Theseus/task_theseus.py \
  --do_train='True' \
  --do_eval='True' \
  --serve_for_online='True' \
  --model=Glue_${dataset} \
  --task_name=${dataset}  \
  --gpu_num_per_node=1 \
  --num_epochs=${epoch} \
  --train_data_dir=$train_data_dir \
  --train_example_num=$train_example_num \
  --eval_data_dir=$eval_data_dir \
  --eval_example_num=$eval_example_num \
  --teacher_model=${FT_BERT_BASE_DIR} \
  --student_model=${TMP_STUDENT_DIR} \
  --batch_size_per_device=8 \
  --eval_batch_size_per_device=16 \
  --loss_print_every_n_iter 10 \
  --log_dir=${TRAIN_LOG_DIR} \
  --result_dir=${RESULT_DIR} \
  --model_save_dir=${STUDENT_DIR} \
  --seq_length=128 \
  --num_hidden_layers=12 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --hidden_size_per_head=64 \
  --hidden_size=768 \
  --learning_rate=$learning_rate \
  --model_save_every_n_iter=50000 \
  --weight_decay_rate=$wd \
  --replacing_rate=${REPLACING_RATE} \
  --student_num_hidden_layers=${STUDENT_NUM_HIDDEN_LAYERS}
