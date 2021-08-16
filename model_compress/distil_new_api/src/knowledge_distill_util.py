"""
Copyright 2020 Tianshu AI Platform. All Rights Reserved.
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
import math
import oneflow.experimental as flow
import oneflow.typing as tp
from typing import Tuple,Any

def soft_cross_entropy(predicts, targets):
    student_likelihood = predicts
    targets_prob = targets
    tmp = flow.negative(targets_prob)*student_likelihood
    res = flow.mean(tmp)
    return res

def mseloss(rep1, rep2):
    return flow.mean(flow.square(rep1-rep2))

def layer_distill(args, student_reps, teacher_reps):
    rep_loss = 0.
    teacher_layer_num = len(teacher_reps) - 1
    student_layer_num = len(student_reps) - 1

    assert teacher_layer_num % student_layer_num == 0
    layers_per_block = int(teacher_layer_num / student_layer_num)

    new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]
    new_student_reps = student_reps

    for student_rep, teacher_rep in zip(new_student_reps, new_teacher_reps):
        tmp_loss = mseloss(student_rep, teacher_rep)
        rep_loss += tmp_loss
    return rep_loss


def att_distill(args, student_atts, teacher_atts):
    att_loss = 0.
    teacher_layer_num = len(teacher_atts)
    student_layer_num = len(student_atts)

    assert teacher_layer_num % student_layer_num == 0
    layers_per_block = int(teacher_layer_num / student_layer_num)
    new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1] for i in range(student_layer_num)]

    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        student_att = flow.where(student_att <= -1e2, flow.zeros(student_att.shape).to('cuda'), student_att)
        teacher_att = flow.where(teacher_att <= -1e2, flow.zeros(teacher_att.shape).to('cuda'), teacher_att)

        tmp_loss = mseloss(student_att, teacher_att)
        att_loss += tmp_loss

    return att_loss

def pred_distill(args, student_logits, teacher_logits):
    soft_loss = soft_cross_entropy(student_logits / args.temperature,
                                  teacher_logits / args.temperature)
    return soft_loss

