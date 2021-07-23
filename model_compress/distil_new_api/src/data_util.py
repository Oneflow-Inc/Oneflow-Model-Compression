import oneflow.experimental as flow
import os
class OFRecordDataLoader(object):
    def __init__(self,data_dir, batch_size, data_part_num, seq_length, part_name_prefix, dataset_size, shuffle=True):
        self.train_record_reader= flow.nn.OfrecordReader(data_dir,
                                             batch_size=batch_size,
                                             data_part_num=data_part_num,
                                             part_name_prefix=part_name_prefix,
                                             random_shuffle=shuffle,
                                             shuffle_after_epoch=shuffle)
        self.dataset_size =  dataset_size
        self.batch_size = batch_size                                    
        self.input_ids_decoder = flow.nn.OfrecordRawDecoder("input_ids", [seq_length], dtype=flow.int32)
        self.input_mask_decoder = flow.nn.OfrecordRawDecoder("input_mask", [seq_length], dtype=flow.int32)
        self.segment_ids_decoder = flow.nn.OfrecordRawDecoder("segment_ids", [seq_length], dtype=flow.int32)
        self.label_ids_decoder = flow.nn.OfrecordRawDecoder("label_ids", [1], dtype=flow.int32)
        self.is_real_example_decoder = flow.nn.OfrecordRawDecoder("is_real_example", [1], dtype=flow.int32)

    def __len__(self):
        return self.dataset_size // self.batch_size

    def get_batch(self):
        train_record = self.train_record_reader()
        input_ids = self.input_ids_decoder(train_record)
        input_mask = self.input_mask_decoder(train_record)
        segment_ids = self.segment_ids_decoder(train_record)
        label_ids = self.label_ids_decoder(train_record)
        is_real_example = self.is_real_example_decoder(train_record)
        blob_confs = {"input_ids":input_ids,"input_mask":input_mask,"segment_ids":segment_ids,"label_ids":label_ids,"is_real_example":is_real_example}

        return blob_confs

if __name__ == "__main__":
    data_dir = '/remote-home/rpluo/Oneflow-Model-Compression/model_compress/data/glue_ofrecord_test/SST-2/train/'
    batch_size = 16
    data_part_num =  1
    seq_length = 128
    part_name_prefix = 'train.of_record-'
    shuffle=True
    flow.enable_eager_execution()
    flow.InitEagerGlobalSession()
    dataloader = OFRecordDataLoader(data_dir,batch_size,data_part_num,seq_length,part_name_prefix,67349)
    result = dataloader.get_batch()
    print(result)