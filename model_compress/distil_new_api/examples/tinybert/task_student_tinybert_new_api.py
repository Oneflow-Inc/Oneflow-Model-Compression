from typing import Sequence
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
from knowledge_distill_util import layer_distill, pred_distill,att_distill
def _parse_args():

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Unsupported value encountered.')

    parser = configs.get_parser()
    parser.add_argument("--task_name", type=str, default='CoLA')
    parser.add_argument("--teacher_model", default=None, type=str, help="The teacher model dir.")
    parser.add_argument("--student_model", default=None, type=str, help="The student model dir.")
    parser.add_argument("--total_model", default=None, type=str, help="The student model dir.")

    parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs')
    parser.add_argument("--train_data_dir", type=str, default='/remote-home/rpluo/oneflow-model-compression/model_compress/data/glue_ofrecord_test/SST-2/train/')
    parser.add_argument("--train_data_prefix", type=str, default='train.of_record-')
    parser.add_argument("--train_example_num", type=int, default=67349,
                        help="example number in dataset")
    parser.add_argument("--batch_size_per_device", type=int, default=8)
    parser.add_argument("--train_data_part_num", type=int, default=1,
                        help="data part number in dataset")
    parser.add_argument("--eval_data_dir", type=str, default='/remote-home/rpluo/oneflow-model-compression/model_compress/data/glue_ofrecord_test/SST-2/eval/')
    parser.add_argument("--eval_data_prefix", type=str, default='eval.of_record-')
    parser.add_argument("--eval_example_num", type=int, default=872,
                        help="example number in dataset")
    parser.add_argument("--eval_batch_size_per_device", type=int, default=12)
    parser.add_argument("--eval_data_part_num", type=int, default=1,
                        help="data part number in dataset")
    parser.add_argument("--result_dir", type=str, default="", help="the save directory of results")

    #
    parser.add_argument("--student_num_hidden_layers", type=int, default=3)
    parser.add_argument("--student_num_attention_heads", type=int, default=12)
    parser.add_argument("--student_max_position_embeddings", type=int, default=512)
    parser.add_argument("--student_type_vocab_size", type=int, default=2)
    parser.add_argument("--student_vocab_size", type=int, default=30522)
    parser.add_argument("--student_attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--student_hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--student_hidden_size_per_head", type=int, default=64)
    parser.add_argument("--student_hidden_size", type=int, default=768)

    parser.add_argument("--teacher_num_hidden_layers", type=int, default=12)
    parser.add_argument("--teacher_num_attention_heads", type=int, default=12)
    parser.add_argument("--teacher_max_position_embeddings", type=int, default=512)
    parser.add_argument("--teacher_type_vocab_size", type=int, default=2)
    parser.add_argument("--teacher_vocab_size", type=int, default=30522)
    parser.add_argument("--teacher_attention_probs_dropout_prob", type=float, default=0.1)
    parser.add_argument("--teacher_hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--teacher_hidden_size_per_head", type=int, default=64)
    parser.add_argument("--teacher_hidden_size", type=int, default=768)

    parser.add_argument("--kd_alpha", type=float, default=0.2)
    parser.add_argument("--kd_beta", type=float, default=10, help='the proposed loss {10,100,500,1000}')
    parser.add_argument('--from_scratch',  type=str2bool, nargs='?', const=False, help='train the student model from scratch or initialize from teacher layers')
    
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--aug_train', type=str2bool, nargs='?', const=False, help='using augmented training set?')

    parser.add_argument('--serve_for_online', type=str2bool, nargs='?', const=False,
                        help='if serve for online, then after training, will delete the teacher params and optimizer parmas from model_save_dir')


    return parser.parse_args()



class TinyBERT(nn.Module):
    def __init__(self,  student_vocab_size, 
                        student_hidden, 
                        student_n_layers, 
                        student_attn_heads, 
                        student_dropout,
                        teacher_vocab_size,
                        teacher_hidden,
                        teacher_n_layers,
                        teacher_attn_heads,
                        teacher_dropout):
        super().__init__()
        self.student_model = BERT(student_vocab_size, 
                                  student_hidden, 
                                  student_n_layers, 
                                  student_attn_heads, 
                                  student_dropout)
        self.teacher_model = BERT(teacher_vocab_size,
                                  teacher_hidden,
                                  teacher_n_layers,
                                  teacher_attn_heads,
                                  teacher_dropout)
        self.student_output_layer = nn.Linear(student_hidden,2)
        self.teacher_output_layer = nn.Linear(teacher_hidden,2)
        self.student_softmax = nn.Softmax(dim=1)
        self.teacher_softmax = nn.Softmax(dim=1)
    def eval_forward(self, x, segment_info):
        student_output,_,_ = self.student_model(x,segment_info)
        student_output2 = self.student_output_layer(student_output[:,0])
        student_logits = self.student_softmax(student_output2)
        return student_logits

    def forward(self, x, segment_info):
        student_output,student_sequence_out,student_att_out = self.student_model(x,segment_info)
        student_output2 = self.student_output_layer(student_output[:,0])
        student_logits = self.student_softmax(student_output2)

        teacher_output,teacher_sequence_out,teacher_att_out = self.teacher_model(x,segment_info)
        teacher_output2 = self.teacher_output_layer(teacher_output[:,0])
        teacher_logits = self.teacher_softmax(teacher_output2)
        return student_logits, student_sequence_out, student_att_out,teacher_logits, teacher_sequence_out, teacher_att_out

def eval(model, dataloader, desc = "train"):
    model.eval()
    labels = []
    predictions = []
    start_time = time.time()
    with flow.no_grad():
        for b in tqdm(range(len(dataloader))):
            blob_confs = dataloader.get_batch()
            input_ids = blob_confs['input_ids'].to("cuda")
            segment_ids = blob_confs['segment_ids'].to("cuda")
            label_ids = blob_confs['label_ids'].squeeze(-1)
            student_logits = model.eval_forward(input_ids, segment_ids)
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

    acc_tasks = ["mnli", "mrpc", "sst-2", "qqp", "qnli", "rte"]
    corr_tasks = ["sts-b"]
    mcc_tasks = ["cola"]

    task_name = args.task_name.lower()
    flow.enable_eager_execution()
    flow.InitEagerGlobalSession()

    train_data_loader = OFRecordDataLoader( args.train_data_dir,
                                            args.batch_size_per_device,
                                            args.train_data_part_num,
                                            args.seq_length,
                                            args.train_data_prefix,
                                            args.train_example_num)
    
    eval_data_loader = OFRecordDataLoader(args.eval_data_dir,
                                            args.eval_batch_size_per_device,
                                            args.eval_data_part_num,
                                            args.seq_length,
                                            args.eval_data_prefix,
                                            args.eval_example_num)
    model = TinyBERT(
                    args.student_vocab_size,
                    args.student_hidden_size,
                    args.student_num_hidden_layers,
                    args.student_num_attention_heads,
                    args.student_hidden_dropout_prob,
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
        of_sgd = flow.optim.SGD(
            model.parameters(), lr=args.learning_rate)
        of_losses = []
        all_samples = len(eval_data_loader) * args.eval_batch_size_per_device
        print_interval = 10
        best_dev_acc = 0.0

        for epoch in range(args.num_epochs):
            model.train()
            for b in range(len(train_data_loader)):
                blob_confs = train_data_loader.get_batch()

                # oneflow train
                start_t = time.time()
                input_ids = blob_confs['input_ids'].to("cuda")
                segment_ids = blob_confs['segment_ids'].to("cuda")
                label_ids = blob_confs['label_ids'].squeeze(-1).to("cuda")

                student_logits, student_sequence_out,student_atts, teacher_logits, teacher_sequence_out,teacher_atts = model(input_ids, segment_ids)

                rep_loss = layer_distill(args, student_sequence_out, teacher_sequence_out)
                att_loss = att_distill(args, student_atts, teacher_atts)
                cls_loss = pred_distill(args, student_logits, teacher_logits)


                loss = rep_loss+att_loss+cls_loss
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
            # eval(model,train_data_loader,desc = 'train')
            print('EvalValJob...')
            result = eval(model,eval_data_loader,desc = 'eval')

            save_model = False
            if task_name in acc_tasks and result['accuracy'] > best_dev_acc:
                best_dev_acc = result['accuracy']
                save_model = True

            if task_name in corr_tasks and result['corr'] > best_dev_acc:
                best_dev_acc = result['corr']
                save_model = True

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
        result = eval(model,eval_data_loader,desc = 'eval')


if __name__ == "__main__":
    args = _parse_args()
    main(args)