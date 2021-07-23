# Copyright (c) The Tianshu Platform Authors.
# Licensed under the Apache License

# which GPU to use
GPU=$1

dataset=$2

STUDENT_NAME=$3

# ofrecord dataset dir
DATA_ROOT=/usr/local/glue_ofrecord

train_data_dir=$DATA_ROOT/${dataset}/train
eval_data_dir=$DATA_ROOT/${dataset}/eval

# saved student model dir
STUDENT_DIR="./models/student_model/${dataset}/${STUDENT_NAME}"

#model_load_dir=./log/${dataset}_bert_theseus_uncased_L-12_H-768_A-12_oneflow_v1/snapshot_last_snapshot


if [ $dataset = "CoLA" ]; then
  train_example_num=8551
  eval_example_num=1043
  test_example_num=1063
  epoch=1
  wd=0.0001
elif [ $dataset = "MRPC" ]; then
  train_example_num=3668
  eval_example_num=408
  test_example_num=1725
  epoch=1
  wd=0.0001
elif [ $dataset = "SST-2" ]; then
  train_example_num=67349
  eval_example_num=872
  test_example_num=1821
elif [ $dataset = "QQP" ]; then
  train_example_num=363849
  eval_example_num=40430
  test_example_num=0
  learning_rate=2e-5
  epoch=1
  wd=0.0001
elif [ $dataset = "MNLI" ]; then
  train_example_num=392702
  eval_example_num=9815
  test_example_num=0
  learning_rate=2e-5
  epoch=1
  wd=0.0001
elif [ $dataset = "WNLI" ]; then
  train_example_num=635
  eval_example_num=71
  test_example_num=0
  learning_rate=2e-5
  epoch=1
  wd=0.0001
elif [ $dataset = "RTE" ]; then
  train_example_num=2490
  eval_example_num=277
  test_example_num=0
  learning_rate=2e-5
  epoch=1
  wd=0.0001
else
  echo "dataset must be GLUE such as 'CoLA','MRPC','SST-2','QQP','MNLI','WNLI','STS-B',"
  exit
fi

# mkdir -p ${model_save_dir}

replace_prob=1.0

CUDA_VISIBLE_DEVICES=${GPU} python3 ./theseus/run_classifier.py \
  --do_train=false \
  --do_eval=True \
  --model=Glue_$dataset \
  --task_name=$dataset  \
  --gpu_num_per_node=1 \
  --num_epochs=$epoch \
  --train_data_dir=$train_data_dir \
  --train_example_num=$train_example_num \
  --eval_data_dir=$eval_data_dir \
  --eval_example_num=$eval_example_num \
  --model_load_dir=${STUDENT_DIR} \
  --batch_size_per_device=32 \
  --eval_batch_size_per_device=4 \
  --loss_print_every_n_iter 20 \
  --log_dir=./log \
  --save_last_snapshot=True \
  --seq_length=128 \
  --num_hidden_layers=4 \
  --num_attention_heads=12 \
  --max_position_embeddings=512 \
  --type_vocab_size=2 \
  --vocab_size=30522 \
  --attention_probs_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --hidden_size_per_head=64 \
  --compress_ratio $compress_ratio \
  --replace_prob $replace_prob \
