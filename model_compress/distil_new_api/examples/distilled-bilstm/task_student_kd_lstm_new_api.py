import oneflow.experimental as flow
import argparse
import numpy as np
import os
import time
import sys
import oneflow.experimental.nn as nn
import json
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "model_compress/distil_new_api/src")))
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "./src")))
import config as configs
from data_util import OFRecordDataLoader
from bert_model.bert import BERT
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
from util import getdirsize
from knowledge_distill_util import pred_distill
from lstm import LSTM
import math
def _parse_args():

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = configs.get_parser()
    parser.add_argument("--task_name", type=str, default='SST-2')
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument("--student_model", default=None, type=str, help="The student model dir.")
    parser.add_argument("--total_model", default=None, type=str, help="The student model dir.")

    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument("--train_data_dir", type=str, default='/remote-home/rpluo/oneflow-model-compression/model_compress/data/glue_ofrecord_test/SST-2/train/')
    parser.add_argument("--train_data_dir_lstm", type=str, default='/remote-home/rpluo/oneflow-model-compression/model_compress/data/glue_ofrecord/SST-2_lstm_32/train')
    parser.add_argument("--train_data_prefix", type=str, default='train.of_record-')
    parser.add_argument("--train_example_num", type=int, default=67349,
                        help="example number in dataset")
    parser.add_argument("--batch_size_per_device", type=int, default=8)
    parser.add_argument("--train_data_part_num", type=int, default=1,
                        help="data part number in dataset")
    parser.add_argument("--eval_data_dir", type=str, default='/remote-home/rpluo/oneflow-model-compression/model_compress/data/glue_ofrecord_test/SST-2/eval')
    parser.add_argument("--eval_data_dir_lstm", type=str, default='/remote-home/rpluo/oneflow-model-compression/model_compress/data/glue_ofrecord/SST-2_lstm_32/eval')
    parser.add_argument("--eval_data_prefix", type=str, default='eval.of_record-')
    parser.add_argument("--eval_example_num", type=int, default=832,
                        help="example number in dataset")
    parser.add_argument("--eval_batch_size_per_device", type=int, default=64)
    parser.add_argument("--eval_data_part_num", type=int, default=1,
                        help="data part number in dataset")
    parser.add_argument("--result_dir", type=str, default="", help="the save directory of results")

    #
    parser.add_argument("--student_num_hidden_layers", type=int, default=24)
    parser.add_argument("--student_num_attention_heads", type=int, default=16)
    parser.add_argument("--student_max_position_embeddings", type=int, default=512)
    parser.add_argument("--student_type_vocab_size", type=int, default=2)
    parser.add_argument("--student_vocab_size", type=int, default=30522)
    parser.add_argument("--student_attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--student_hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--student_hidden_size_per_head", type=int, default=64)
    parser.add_argument("--student_hidden_size", type=int, default=300)
    parser.add_argument("--student_seq_length", type=int, default=32, help="the max seq length for student")

    parser.add_argument("--teacher_num_hidden_layers", type=int, default=24)
    parser.add_argument("--teacher_num_attention_heads", type=int, default=16)
    parser.add_argument("--teacher_max_position_embeddings", type=int, default=512)
    parser.add_argument("--teacher_type_vocab_size", type=int, default=2)
    parser.add_argument("--teacher_vocab_size", type=int, default=30522)
    parser.add_argument("--teacher_attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--teacher_hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--teacher_hidden_size_per_head", type=int, default=64)
    parser.add_argument("--teacher_hidden_size", type=int, default=768)

    parser.add_argument("--kd_alpha", type=float, default=0.1)

    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--aug_train',  type=str2bool, nargs='?', const=False, help='using augmented training set?')

    parser.add_argument('--serve_for_online',  type=str2bool, nargs='?', const=False, help='if serve for online, then after training, will delete the teacher params and optimizer parmas from model_save_dir')

    args = parser.parse_args()

    task_name = args.task_name.lower()

    if args.aug_train:
        args.train_data_dir = args.train_data_dir.replace('train','train_aug')

    batch_size = args.num_nodes * args.gpu_num_per_node * args.batch_size_per_device
    eval_batch_size = args.num_nodes * args.gpu_num_per_node * args.eval_batch_size_per_device

    epoch_size = math.ceil(args.train_example_num / batch_size)
    num_eval_steps = math.ceil(args.eval_example_num / eval_batch_size)
    args.iter_num = epoch_size * args.num_epochs
    configs.print_args(args)

    return parser.parse_args()
def soft_cross_entropy(predicts, targets):
    student_likelihood = predicts
    targets_prob = targets
    tmp = flow.negative(targets_prob)*student_likelihood
    res = flow.mean(tmp)
    return res
def pred_distill(args, student_logits, teacher_logits):
    soft_loss = soft_cross_entropy(student_logits / args.temperature,
                                  teacher_logits / args.temperature)
    return soft_loss

def getdirsize(dir):
    size = 0
    for root, dirs, files in os.walk(dir):
        for name in files:
            if str(root[-2:]) == '-v' or str(root[-2:]) == '-m':
                pass
            else:
                tmp = os.path.getsize(os.path.join(root, name))
                size += tmp
        # size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    return size



class kd_lstm_model(nn.Module):
    def __init__(self,  student_vocab_size, 
                        student_hidden, 
                        student_n_layers, 
                        student_attn_heads, 
                        student_dropout,
                        intermediate_size,
                        teacher_vocab_size,
                        teacher_hidden,
                        teacher_n_layers,
                        teacher_attn_heads,
                        teacher_dropout):
        super().__init__()
        self.student_model = LSTM(student_vocab_size,
                                    student_hidden, 
                                    student_hidden, 
                                    intermediate_size)
        self.teacher_model = BERT(teacher_vocab_size,
                                  teacher_hidden,
                                  teacher_n_layers,
                                  teacher_attn_heads,
                                  teacher_dropout)
        self.student_output_layer = nn.Linear(intermediate_size,2)
        self.teacher_output_layer = nn.Linear(teacher_hidden,2)
        self.student_softmax = nn.Softmax(dim=1)
        self.teacher_softmax = nn.Softmax(dim=1)
    def eval_forward(self, x_lstm):
        student_output = self.student_model(x_lstm)
        student_output2 = self.student_output_layer(student_output)
        student_logits = self.student_softmax(student_output2)
        return student_logits
    def forward(self, x_lstm, x, segment_info):
        student_output = self.student_model(x_lstm)
        student_output2 = self.student_output_layer(student_output)
        student_logits = self.student_softmax(student_output2)

        teacher_output,_,_ = self.teacher_model(x,segment_info)
        teacher_output2 = self.teacher_output_layer(teacher_output[:,0])
        teacher_logits = self.teacher_softmax(teacher_output2)
        return student_logits, teacher_logits

def eval(model, lstm_dataloader, desc = "train"):
    model.eval()
    labels = []
    predictions = []
    start_time = time.time()
    with flow.no_grad():
        for b in tqdm(range(len(lstm_dataloader))):
            blob_confs_lstm = lstm_dataloader.get_batch()
            input_ids = blob_confs_lstm['input_ids'].to("cuda")
            label_ids = blob_confs_lstm['label_ids'].squeeze(-1)
            student_logits = model.eval_forward(input_ids)
            predictions.extend(student_logits.detach().to('cpu').numpy().argmax(axis=1).tolist())
            labels.extend(label_ids.tolist())
    end_time = time.time()
    cost_time = end_time - start_time
    print('cost time: {} s'.format(cost_time))

    model_size = getdirsize(args.model_save_dir)
    print('model_size: %d Mbytes' % (model_size / 1024 / 1024))  # Mbytes

    accuracy = accuracy_score(labels, predictions)
    mcc = matthews_corrcoef(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f_1 = f1_score(labels, predictions)
    save_dict = {"accuracy": "%.2f" % accuracy,
                 "MCC": "%.2f" % mcc,
                 "precision": "%.2f" % precision,
                 "recall": "%.2f" % recall,
                 "f_1": "%.2f" % f_1,
                 "modelSize": "%d" % (model_size / 1024 / 1024),
                 "reasoningTime": "%.2f" % (args.eval_example_num / cost_time)}  # sample/second

    if args.result_dir == "":
        args.result_dir = args.model_save_dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    with open(os.path.join(args.result_dir, 'results_{}.json'.format(desc)), "w") as f:
        json.dump(save_dict, f)

    def metric_fn(predictions, labels):
        return {
            "accuracy": accuracy,
            "matthews_corrcoef": mcc,
            "precision": precision,
            "recall": recall,
            "f1": f_1,
        }

    metric_dict = metric_fn(predictions, labels)
    print(desc, ', '.join('{}: {:.3f}'.format(k, v) for k, v in metric_dict.items()))
    return metric_dict      

def main(args):
    glue_output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
}

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    task_name = args.task_name.lower()
    flow.enable_eager_execution()
    flow.InitEagerGlobalSession()

    train_data_loader_lstm = OFRecordDataLoader( args.train_data_dir_lstm,
                                            args.batch_size_per_device,
                                            args.train_data_part_num,
                                            args.student_seq_length,
                                            args.train_data_prefix,
                                            args.train_example_num,
                                            False)
    train_data_loader = OFRecordDataLoader( args.train_data_dir,
                                            args.batch_size_per_device,
                                            args.train_data_part_num,
                                            args.seq_length,
                                            args.train_data_prefix,
                                            args.train_example_num,
                                            False)
    eval_data_loader_lstm = OFRecordDataLoader(args.eval_data_dir_lstm,
                                            args.eval_batch_size_per_device,
                                            args.eval_data_part_num,
                                            args.student_seq_length,
                                            args.eval_data_prefix,
                                            args.eval_example_num,
                                            False)

    eval_data_loader = OFRecordDataLoader(args.eval_data_dir,
                                            args.eval_batch_size_per_device,
                                            args.eval_data_part_num,
                                            args.seq_length,
                                            args.eval_data_prefix,
                                            args.eval_example_num,
                                            False)
    intermediate_size = 400

    model = kd_lstm_model(
                    args.student_vocab_size,
                    args.student_hidden_size,
                    args.student_num_hidden_layers,
                    args.student_num_attention_heads,
                    args.student_hidden_dropout_prob,
                    intermediate_size,
                    args.teacher_vocab_size,
                    args.teacher_hidden_size,
                    args.teacher_num_hidden_layers,
                    args.teacher_num_attention_heads,
                    args.teacher_hidden_dropout_prob
                     )

    model.to('cuda')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)
    if args.do_train:
        of_cross_entropy = flow.nn.CrossEntropyLoss(reduction='mean')
        of_cross_entropy.to("cuda")
        of_sgd = flow.optim.Adam(
            model.parameters(), lr=args.learning_rate)
        of_losses = []
        all_samples = len(eval_data_loader) * args.eval_batch_size_per_device
        print_interval = 10
        print('start training......')
        best_dev_acc = 0.0
        for epoch in range(args.num_epochs):
            model.train()
            for b in range(len(train_data_loader)):
                blob_confs = train_data_loader.get_batch()
                lstm_blob_confs = train_data_loader_lstm.get_batch()
                # oneflow train
                start_t = time.time()
                input_ids = blob_confs['input_ids'].to("cuda")
                segment_ids = blob_confs['segment_ids'].to("cuda")
                label_ids = blob_confs['label_ids'].squeeze(-1).to("cuda")

                input_ids_lstm = lstm_blob_confs['input_ids'].to('cuda')
                
                student_logits, teacher_logits = model(input_ids_lstm,input_ids, segment_ids)

                cls_loss = pred_distill(args, student_logits, teacher_logits)

                loss_ce = of_cross_entropy(student_logits, label_ids)

                loss = loss_ce * args.kd_alpha + cls_loss * (1 - args.kd_alpha)
                loss.backward()
                of_sgd.step()
                of_sgd.zero_grad()
                end_t = time.time()
                if b % print_interval == 0:
                    l = loss.numpy()[0]
                    of_losses.append(l)
                    print(
                        "epoch {} train iter {} oneflow loss {}, train time : {}".format(
                            epoch, b, l, end_t - start_t
                        )
                    )
            # print('EvalTrainJob...')
            # eval(model,train_data_loader_lstm,desc = 'train')
                print('EvalValJob...')
                result = eval(model,eval_data_loader_lstm,desc = 'eval')

                save_model = False
                if task_name in acc_tasks and result['accuracy'] > best_dev_acc:
                    best_dev_acc = result['accuracy']
                    save_model = True

                # if task_name in corr_tasks and result['corr'] > best_dev_acc:
                #     best_dev_acc = result['corr']
                #     save_model = True

                if task_name in mcc_tasks and result['matthews_corrcoef'] > best_dev_acc:
                    best_dev_acc = result['matthews_corrcoef']
                    save_model = True
                    print('Best result:', result)

                if save_model:
                    if os.path.exists(args.model_save_dir):
                        import shutil
                        shutil.rmtree(args.model_save_dir)
                    if not os.path.exists(args.model_save_dir):
                        os.makedirs(args.model_save_dir)
                    snapshot_save_path = os.path.join(args.model_save_dir)
                    print("Saving best model to {}".format(snapshot_save_path))
                    flow.save(model.state_dict(),snapshot_save_path)


    if args.do_eval:
        print('Loading model...')
        print(args.model_save_dir)

        if not args.do_train:
            model_dict = flow.load(args.model_save_dir)
            print('successful')
            model.load_state_dict(model_dict)
        print('Evaluation...')
        result = eval(model,eval_data_loader_lstm,desc = 'eval')


if __name__ == "__main__":
    args = _parse_args()
    main(args)