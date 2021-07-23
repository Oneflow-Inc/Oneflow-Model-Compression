import oneflow as flow
@flow.global_function(type="predict")
def get_data():
    data_dir = '/remote-home/rpluo/Oneflow-Model-Compression/model_compress/data/glue_ofrecord_test/SST-2/train/'
    batch_size = 16
    data_part_num =  1
    seq_length = 128
    part_name_prefix = 'train.of_record-'
    shuffle = True
    with flow.scope.placement("cpu", "0:0"):
        ofrecord = flow.data.ofrecord_reader(data_dir,
                                             batch_size=batch_size,
                                             data_part_num=data_part_num,
                                             part_name_prefix=part_name_prefix,
                                             random_shuffle=shuffle,
                                             shuffle_after_epoch=shuffle)
        blob_confs = {}

        def _blob_conf(name, shape, dtype=flow.int32):
            blob_confs[name] = flow.data.OFRecordRawDecoder(ofrecord, name, shape=shape, dtype=dtype)

        _blob_conf("input_ids", [seq_length])
        _blob_conf("input_mask", [seq_length])
        _blob_conf("segment_ids", [seq_length])
        _blob_conf("label_ids", [1])
        _blob_conf("is_real_example", [1])
    return blob_confs


if __name__ == "__main__":
    
    train_data = get_data()
    print(train_data)