"""
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import time
import numpy as np
import pandas as pd
from datetime import datetime
import oneflow as flow
<<<<<<< HEAD
import json

=======
import config as configs
import json

parser = configs.get_parser()
args = parser.parse_args()

>>>>>>> tianshu

def InitNodes(args):
    if args.num_nodes > 1:
        assert args.num_nodes <= len(args.node_ips)
        flow.env.ctrl_port(args.ctrl_port)
        nodes = []
        for ip in args.node_ips[:args.num_nodes]:
            addr_dict = {}
            addr_dict["addr"] = ip
            nodes.append(addr_dict)

        flow.env.machine(nodes)


class Snapshot(object):
    def __init__(self, model_save_dir, model_load_dir):
        self._model_save_dir = model_save_dir
        self._check_point = flow.train.CheckPoint()
        if model_load_dir:
            assert os.path.isdir(model_load_dir)
            print("Restoring model from {}.".format(model_load_dir))
            self._check_point.load(model_load_dir)
        else:
            self._check_point.init()
            self.save('initial_model')
            print("Init model on demand.")

    def save(self, name):
        snapshot_save_path = os.path.join(self._model_save_dir, "snapshot_{}".format(name))
        if not os.path.exists(snapshot_save_path):
            os.makedirs(snapshot_save_path)
        print("Saving model to {}.".format(snapshot_save_path))
        self._check_point.save(snapshot_save_path)


class Summary(object):
    def __init__(self, log_dir, config, modelSize, filename='summary.csv'):
        self._filename = filename
        self._log_dir = log_dir
        self.modelSize = modelSize
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        self._metrics = None
        # self._metrics = pd.DataFrame(
<<<<<<< HEAD
            # {"epoch":0, "iter": 0, "legend": "cfg", "note": str(config)}, 
            # {"epoch": epoch, "iter": step, "legend": legend, "value": value, "rank": 0}
            # index=[0])

            # {"top_1_accuracy": "0.48", "top_k_accuracy": "0.95", "top_k": "6", "modelSize": "0", "reasoningTime": "2238.79"}
=======
        # {"epoch":0, "iter": 0, "legend": "cfg", "note": str(config)},
        # {"epoch": epoch, "iter": step, "legend": legend, "value": value, "rank": 0}
        # index=[0])

        # {"top_1_accuracy": "0.48", "top_k_accuracy": "0.95", "top_k": "6", "modelSize": "0", "reasoningTime": "2238.79"}
>>>>>>> tianshu

    def scalar(self, top_1_accuracy, top_k_accuracy, top_k, reasoningTime):
        # TODO: support rank(which device/gpu)
        df = pd.DataFrame({
<<<<<<< HEAD
            "top_1_accuracy": top_1_accuracy, 
            "top_k_accuracy": top_k_accuracy, 
            "top_k": top_k, 
            "modelSize": self.modelSize, 
            "reasoningTime": reasoningTime, 
            }, index=[0])
=======
            "accuracy": "%.2f" % top_1_accuracy,
            "top_k_accuracy": "%.2f" % top_k_accuracy,
            "top_k": "%d" % top_k,
            "modelSize": "%d" % (self.modelSize / 1024 / 1024),
            "reasoningTime": "%.2f" % reasoningTime
        }, index=[0])
>>>>>>> tianshu
        if self._metrics is None:
            self._metrics = df
        else:
            self._metrics = pd.concat([self._metrics, df], axis=0, sort=False)

    def save(self):
        save_path = os.path.join(self._log_dir, self._filename)
        # import numpy as np
        # self._metrics.to_csv(save_path, index=False)
<<<<<<< HEAD
        ret = self._metrics.max(axis=0).to_json()
        with open('ret.json', 'w') as f:
            json.dump(ret, f)

=======
        ret = self._metrics.max(axis=0)
        ret = {
            'accuracy': ret[0],
            'top_k_accuracy': ret[1],
            'top_k': ret[2],
            'modelSize': ret[3],
            'reasoningTime': ret[4]
        }
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        with open(os.path.join(args.result_dir, "results_eval.json"), "w") as f:
            json.dump(ret, f)


>>>>>>> tianshu
class StopWatch(object):
    def __init__(self):
        pass

    def start(self):
        self.start_time = time.time()
        self.last_split = self.start_time

    def split(self):
        now = time.time()
        duration = now - self.last_split
        self.last_split = now
        return duration

    def stop(self):
        self.stop_time = time.time()

    def duration(self):
        return self.stop_time - self.start_time


def match_top_k(predictions, labels, top_k=1):
    max_k_preds = np.argpartition(predictions.numpy(), -top_k)[:, -top_k:]
    match_array = np.logical_or.reduce(max_k_preds == labels.reshape((-1, 1)), axis=1)
    num_matched = match_array.sum()
    return num_matched, match_array.shape[0]


class Metric(object):
    def __init__(self, summary=None, save_summary_steps=-1, desc='train', calculate_batches=-1,
                 batch_size=256, top_k=5, prediction_key='predictions', label_key='labels',
                 loss_key=None):
        self.summary = summary
        self.save_summary = isinstance(self.summary, Summary)
        self.save_summary_steps = save_summary_steps
        self.desc = desc
        self.calculate_batches = calculate_batches
        self.top_k = top_k
        self.prediction_key = prediction_key
        self.label_key = label_key
        self.loss_key = loss_key
        if loss_key:
            self.fmt = "{}: epoch {}, iter {}, loss: {:.6f}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}"
        else:
            self.fmt = "{}: epoch {}, iter {}, top_1: {:.6f}, top_k: {:.6f}, samples/s: {:.3f}"

        self.timer = StopWatch()
        self.timer.start()
        self._clear()

    def _clear(self):
        self.top_1_num_matched = 0
        self.top_k_num_matched = 0
        self.num_samples = 0.0

    def metric_cb(self, epoch, step):
        def callback(outputs):
            if step == 0: self._clear()
            if self.prediction_key:
                num_matched, num_samples = match_top_k(outputs[self.prediction_key],
                                                       outputs[self.label_key])
                self.top_1_num_matched += num_matched
                num_matched, _ = match_top_k(outputs[self.prediction_key],
                                             outputs[self.label_key], self.top_k)
                self.top_k_num_matched += num_matched
            else:
                num_samples = outputs[self.label_key].shape[0]

            self.num_samples += num_samples

            if (step + 1) % self.calculate_batches == 0:
                throughput = self.num_samples / self.timer.split()
                if self.prediction_key:
                    top_1_accuracy = self.top_1_num_matched / self.num_samples
                    top_k_accuracy = self.top_k_num_matched / self.num_samples
                else:
                    top_1_accuracy = 0.0
                    top_k_accuracy = 0.0

                if self.loss_key:
                    loss = outputs[self.loss_key].mean()
                    print(self.fmt.format(self.desc, epoch, step + 1, loss, top_1_accuracy,
                                          top_k_accuracy, throughput), time.time())
                    if self.save_summary:
                        self.summary.scalar(top_1_accuracy, top_k_accuracy, self.top_k, throughput)
                else:
                    print(self.fmt.format(self.desc, epoch, step + 1, top_1_accuracy,
                                          top_k_accuracy, throughput), time.time())

                self._clear()

                # if self.save_summary:
<<<<<<< HEAD
                    # self.summary.scalar(top_1_accuracy, top_k_accuracy, self.top_k, throughput)
=======
                # self.summary.scalar(top_1_accuracy, top_k_accuracy, self.top_k, throughput)
>>>>>>> tianshu

                if self.save_summary:
                    # self.summary.scalar(self.desc + "_throughput", throughput, epoch, step)
                    # self.summary.scalar(top_1_accuracy, top_k_accuracy, self.top_k, throughput)
                    if self.prediction_key:
                        # self.summary.scalar(self.desc + "_top_1", top_1_accuracy, epoch, step)
                        # self.summary.scalar(self.desc + "_top_{}".format(self.top_k),
<<<<<<< HEAD
                                            # top_k_accuracy, epoch, step)
=======
                        # top_k_accuracy, epoch, step)
>>>>>>> tianshu
                        self.summary.scalar(top_1_accuracy, top_k_accuracy, self.top_k, throughput)

            if self.save_summary:
                if (step + 1) % self.save_summary_steps == 0:
                    self.summary.save()

        return callback
<<<<<<< HEAD


=======
>>>>>>> tianshu
