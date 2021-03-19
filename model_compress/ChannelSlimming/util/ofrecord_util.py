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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oneflow as flow


def add_ofrecord_args(parser):
    parser.add_argument("--image_size", type=int, default=224,
                        required=False, help="image size")
    parser.add_argument("--resize_shorter", type=int, default=256,
                        required=False, help="resize shorter for validation")
    parser.add_argument("--train_data_dir", type=str,
                        default=None, help="train dataset directory")
    parser.add_argument("--train_data_part_num", type=int,
                        default=256, help="train data part num")
    parser.add_argument("--val_data_dir", type=str,
                        default=None, help="val dataset directory")
    parser.add_argument("--val_data_part_num", type=int,
                        default=256, help="val data part num")
    return parser

#old version, cancelled
def load_imagenet(args, batch_size, data_dir, data_part_num, codec):
    image_blob_conf = flow.data.BlobConf(
        "encoded",
        shape=(args.image_size, args.image_size, 3),
        dtype=flow.float,
        codec=codec,
        preprocessors=[flow.data.NormByChannelPreprocessor(args.rgb_mean[::-1],
                                                           args.rgb_std[::-1])],
        # preprocessors=[flow.data.NormByChannelPreprocessor(args.rgb_mean, args.rgb_std)], #bgr2rgb
    )

    label_blob_conf = flow.data.BlobConf(
        "class/label", shape=(), dtype=flow.int32, codec=flow.data.RawCodec()
    )

    return flow.data.decode_ofrecord(
        data_dir,
        (label_blob_conf, image_blob_conf),
        batch_size=batch_size,
        data_part_num=data_part_num,
        part_name_suffix_length=5,
        #shuffle = True,
        # buffer_size=32768,
        name="decode")
    
#old version, cancelled
def load_cifar10(data_dir, batch_size, data_part_num, image_size=32):
    image_blob_conf = flow.data.BlobConf(
        "images",
        shape=(image_size, image_size, 3),
        dtype=flow.float,
        codec=flow.data.RawCodec(),
        preprocessors=[flow.data.NormByChannelPreprocessor((125.31, 122.96, 113.86), (61.252, 60.767, 65.852))],
    )
    label_blob_conf = flow.data.BlobConf("labels", shape=(), dtype=flow.int32, codec=flow.data.RawCodec())

    return flow.data.decode_ofrecord(
        data_dir,
        (label_blob_conf, image_blob_conf),
        batch_size=batch_size,
        data_part_num=data_part_num,
        name="decode",
    )


def load_synthetic(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    batch_size = total_device_num * args.batch_size_per_device
    label = flow.data.decode_random(
        shape=(),
        dtype=flow.int32,
        batch_size=batch_size,
        initializer=flow.zeros_initializer(flow.int32),
    )

    image = flow.data.decode_random(
        shape=(args.image_size, args.image_size, 3), dtype=flow.float, batch_size=batch_size
    )

    return label, image


def load_imagenet_for_training(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device

    color_space = 'RGB'
    ofrecord = flow.data.ofrecord_reader(args.train_data_dir,
                                        batch_size=train_batch_size,
                                        data_part_num=args.train_data_part_num,
                                        part_name_suffix_length=5,
                                        random_shuffle=True,
                                        shuffle_after_epoch=True)
    image = flow.data.OFRecordImageDecoderRandomCrop(ofrecord, "encoded",  # seed=seed,
                                                    color_space=color_space)
    label = flow.data.OFRecordRawDecoder(ofrecord, "class/label", shape=(), dtype=flow.int32)
    rsz = flow.image.Resize(image, resize_x=args.image_size, resize_y=args.image_size,
                            color_space=color_space)
    rng = flow.random.CoinFlip(batch_size=train_batch_size)  # , seed=seed)
    normal = flow.image.CropMirrorNormalize(rsz, mirror_blob=rng, color_space=color_space,
                                            mean=args.rgb_mean, std=args.rgb_std, output_dtype=flow.float)
    return label, normal


def load_imagenet_for_validation(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    val_batch_size = total_device_num * args.val_batch_size_per_device

    color_space = 'RGB'
    ofrecord = flow.data.ofrecord_reader(args.val_data_dir,
                                         batch_size=val_batch_size,
                                         data_part_num=args.val_data_part_num,
                                         part_name_suffix_length=5,
                                         shuffle_after_epoch=False)
    image = flow.data.OFRecordImageDecoder(ofrecord, "encoded", color_space=color_space)
    label = flow.data.OFRecordRawDecoder(ofrecord, "class/label", shape=(), dtype=flow.int32)
    rsz = flow.image.Resize(image, resize_shorter=args.resize_shorter, color_space=color_space)
    normal = flow.image.CropMirrorNormalize(rsz, color_space=color_space,
                                            crop_h=args.image_size, crop_w=args.image_size, crop_pos_y=0.5, crop_pos_x=0.5,
                                            mean=args.rgb_mean, std=args.rgb_std, output_dtype=flow.float)
    return label, normal

def load_cifar_for_training(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device

#    color_space = 'RGB'
    ofrecord = flow.data.ofrecord_reader(args.train_data_dir,
                                        batch_size=train_batch_size,
                                        data_part_num=args.train_data_part_num,
                                        part_name_suffix_length=5,
                                        random_shuffle=True,
                                        shuffle_after_epoch=True)
    label = flow.data.OFRecordRawDecoder(ofrecord, "labels", shape=(), dtype=flow.int32)    
    image = flow.data.OFRecordRawDecoder(ofrecord, "images",
                                         shape=(3, args.image_size, args.image_size),
                                         dtype=flow.float)
    image = flow.transpose(image, perm=[0, 2, 3, 1])
    image_uint8 = flow.cast(image, flow.uint8)
    rng = flow.random.CoinFlip(batch_size=train_batch_size)
    normal = flow.image.CropMirrorNormalize(image_uint8, mirror_blob=rng,
                                            mean=args.rgb_mean, std=args.rgb_std)   
    return label, normal

def load_cifar_for_validation(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    val_batch_size = total_device_num * args.val_batch_size_per_device

#    color_space = 'RGB'
    ofrecord = flow.data.ofrecord_reader(args.val_data_dir,
                                            batch_size=val_batch_size,
                                            data_part_num=args.val_data_part_num,
                                            part_name_suffix_length=5,
                                            shuffle_after_epoch=False)
    label = flow.data.OFRecordRawDecoder(ofrecord, "labels", shape=(), dtype=flow.int32)
    image = flow.data.OFRecordRawDecoder(ofrecord, "images",
                                         shape=(3, args.image_size, args.image_size),
                                         dtype=flow.float)
    image = flow.transpose(image, perm=[0, 2, 3, 1])
    image_uint8 = flow.cast(image, flow.uint8)
    normal = flow.image.CropMirrorNormalize(image_uint8, crop_h=args.image_size, crop_w=args.image_size, 
                                            crop_pos_y=0.5, crop_pos_x=0.5,
                                            mean=args.rgb_mean, std=args.rgb_std, output_dtype=flow.float)
    return label, normal

def load_mnist_for_training(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    ofrecord = flow.data.ofrecord_reader(args.train_data_dir,
                                        batch_size=train_batch_size,
                                        data_part_num=args.train_data_part_num,
                                        part_name_suffix_length=5,
                                        random_shuffle=True,
                                        shuffle_after_epoch=True)
    label = flow.data.OFRecordRawDecoder(ofrecord, "labels", shape=(), dtype=flow.int32)    
    image = flow.data.OFRecordRawDecoder(ofrecord, "images",
                                         shape=(1, args.image_size, args.image_size),
                                         dtype=flow.float)
#    print(image.shape)
    image = flow.transpose(image, perm=[0, 2, 3, 1])
    image_uint8 = flow.cast(image, flow.uint8)
    rng = flow.random.CoinFlip(batch_size=train_batch_size)
    normal = flow.image.CropMirrorNormalize(image_uint8, mirror_blob=rng, color_space="GRAY",
                                            mean=args.rgb_mean, std=args.rgb_std)
    return label, normal


def load_mnist_for_validation(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    val_batch_size = total_device_num * args.val_batch_size_per_device
    ofrecord = flow.data.ofrecord_reader(args.val_data_dir,
                                            batch_size=val_batch_size,
                                            data_part_num=args.val_data_part_num,
                                            part_name_suffix_length=5,
                                            shuffle_after_epoch=False)
    label = flow.data.OFRecordRawDecoder(ofrecord, "labels", shape=(), dtype=flow.int32)
    image = flow.data.OFRecordRawDecoder(ofrecord, "images",
                                         shape=(1, args.image_size, args.image_size),
                                         dtype=flow.float)
    image = flow.transpose(image, perm=[0, 2, 3, 1])
    image_uint8 = flow.cast(image, flow.uint8)
    normal = flow.image.CropMirrorNormalize(image_uint8, crop_h=args.image_size, crop_w=args.image_size, 
                                            crop_pos_y=0.5, crop_pos_x=0.5, color_space="GRAY",
                                            mean=args.rgb_mean, std=args.rgb_std, output_dtype=flow.float)
    return label, normal

def load_svhn_for_training(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device

    ofrecord = flow.data.ofrecord_reader(args.train_data_dir,
                                        batch_size=train_batch_size,
                                        data_part_num=args.train_data_part_num,
                                        part_name_suffix_length=5,
                                        random_shuffle=True,
                                        shuffle_after_epoch=True)
    label = flow.data.OFRecordRawDecoder(ofrecord, "labels", shape=(), dtype=flow.int32)    
    image = flow.data.OFRecordRawDecoder(ofrecord, "images",
                                         shape=(args.image_size, args.image_size, 3),
                                         dtype=flow.float)
    image_uint8 = flow.cast(image, flow.uint8)
    rng = flow.random.CoinFlip(batch_size=train_batch_size)
    normal = flow.image.CropMirrorNormalize(image_uint8, mirror_blob=rng,
                                            mean=args.rgb_mean, std=args.rgb_std)   
    return label, normal

def load_svhn_for_validation(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    val_batch_size = total_device_num * args.val_batch_size_per_device

    ofrecord = flow.data.ofrecord_reader(args.val_data_dir,
                                            batch_size=val_batch_size,
                                            data_part_num=args.val_data_part_num,
                                            part_name_suffix_length=5,
                                            shuffle_after_epoch=False)
    label = flow.data.OFRecordRawDecoder(ofrecord, "labels", shape=(), dtype=flow.int32)
    image = flow.data.OFRecordRawDecoder(ofrecord, "images",
                                         shape=(args.image_size, args.image_size, 3),
                                         dtype=flow.float)
    image_uint8 = flow.cast(image, flow.uint8)
    normal = flow.image.CropMirrorNormalize(image_uint8, crop_h=args.image_size, crop_w=args.image_size, 
                                            crop_pos_y=0.5, crop_pos_x=0.5,
                                            mean=args.rgb_mean, std=args.rgb_std, output_dtype=flow.float)
    return label, normal

def load_mydata_for_training(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device

#    color_space = 'RGB'
    ofrecord = flow.data.ofrecord_reader(args.train_data_dir,
                                        batch_size=train_batch_size,
                                        data_part_num=args.train_data_part_num,
                                        part_name_suffix_length=5,
                                        random_shuffle=True,
                                        shuffle_after_epoch=True)
    label = flow.data.OFRecordRawDecoder(ofrecord, "labels", shape=(), dtype=flow.int32)    
    image = flow.data.OFRecordRawDecoder(ofrecord, "images",
                                         shape=(3, args.image_size, args.image_size),
                                         dtype=flow.float)
    image = flow.transpose(image, perm=[0, 2, 3, 1])
    image_uint8 = flow.cast(image, flow.uint8)
    rng = flow.random.CoinFlip(batch_size=train_batch_size)
    normal = flow.image.CropMirrorNormalize(image_uint8, mirror_blob=rng,
                                            mean=args.rgb_mean, std=args.rgb_std)   
    return label, normal

def load_mydata_for_validation(args):
    total_device_num = args.num_nodes * args.gpu_num_per_node
    val_batch_size = total_device_num * args.val_batch_size_per_device

#    color_space = 'RGB'
    ofrecord = flow.data.ofrecord_reader(args.val_data_dir,
                                            batch_size=val_batch_size,
                                            data_part_num=args.val_data_part_num,
                                            part_name_suffix_length=5,
                                            shuffle_after_epoch=False)
    label = flow.data.OFRecordRawDecoder(ofrecord, "labels", shape=(), dtype=flow.int32)
    image = flow.data.OFRecordRawDecoder(ofrecord, "images",
                                         shape=(3, args.image_size, args.image_size),
                                         dtype=flow.float)
    image = flow.transpose(image, perm=[0, 2, 3, 1])
    image_uint8 = flow.cast(image, flow.uint8)
    normal = flow.image.CropMirrorNormalize(image_uint8, crop_h=args.image_size, crop_w=args.image_size, 
                                            crop_pos_y=0.5, crop_pos_x=0.5,
                                            mean=args.rgb_mean, std=args.rgb_std, output_dtype=flow.float)
    return label, normal


if __name__ == "__main__":
    import os
    import config as configs
    from util import Summary, Metric
    from job_function_util import get_val_config
    parser = configs.get_parser()
    args = parser.parse_args()
    configs.print_args(args)

    flow.config.gpu_device_num(args.gpu_num_per_node)
    flow.config.enable_debug_mode(True)
    @flow.global_function(get_val_config(args))
    def IOTest():
        if args.train_data_dir:
            assert os.path.exists(args.train_data_dir)
            print("Loading data from {}".format(args.train_data_dir))
            (labels, images) = load_imagenet_for_training(args)
        else:
            print("Loading synthetic data.")
            (labels, images) = load_synthetic(args)
        outputs = {"images": images, "labels": labels}
        return outputs

    total_device_num = args.num_nodes * args.gpu_num_per_node
    train_batch_size = total_device_num * args.batch_size_per_device
    summary = Summary(args.log_dir, args, filename='io_test.csv')
    metric = Metric(desc='io_test', calculate_batches=args.loss_print_every_n_iter,
                    summary=summary, save_summary_steps=args.loss_print_every_n_iter,
                    batch_size=train_batch_size, prediction_key=None)
    for i in range(1000):
        IOTest().async_get(metric.metric_cb(0, i))
