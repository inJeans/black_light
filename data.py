import os

from tqdm import tqdm

import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt

VIS_DIR = "visible"
UV_DIR = "uv"
IMG_WIDTH = 256
IMG_HEIGHT = 256

IMAGES_PER_TFR = 1024
SHUFFLE_BUFFER_SIZE = 1024
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE

def process_input(tf_example, vis=False, uv=False, image_shape=(IMG_WIDTH, IMG_HEIGHT, 3)):
    input_list = []

    if vis:
        input_list.append(tf.ensure_shape(tf.io.parse_tensor(tf_example["vis"],
                                                             tf.float32),
                                          image_shape)
                         )
    if uv:
        input_list.append(tf.ensure_shape(tf.io.parse_tensor(tf_example["uv"],
                                                             tf.float32),
                                          image_shape)
                         )

    return tuple(input_list), tf.ensure_shape(tf.io.parse_tensor(tf_example["soft_label"],
                                                                 tf.double),
                                              (4,)
                                            )

def prepare_for_training(ds, cache=True, shuffle_buffer_size=SHUFFLE_BUFFER_SIZE, batch_size=BATCH_SIZE):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            print("Creating temp file - %s" % cache)
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    if shuffle_buffer_size is not None:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, 
                        reshuffle_each_iteration=True)

    # Repeat forever
    # ds = ds.repeat()

    ds = ds.batch(batch_size)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=AUTOTUNE)

    return ds

def tf_record_reader(tf_records_dir, vis=False, uv=False):
    list_ds = tf.data.Dataset.list_files(tf_records_dir+"/*",
                                         shuffle=True)
    tfr_ds = tf.data.TFRecordDataset(list_ds, num_parallel_reads=AUTOTUNE)

    image_description_dict = {}
    if vis:
        image_description_dict["vis"] = tf.io.FixedLenFeature([], tf.string)
    if uv:
        image_description_dict["uv"] = tf.io.FixedLenFeature([], tf.string)
    image_description_dict["soft_label"] = tf.io.FixedLenFeature([], tf.string)

    parsed_ds = tfr_ds.map(lambda x: _parse_image_function(x, image_description_dict), 
                           num_parallel_calls=AUTOTUNE)
    
    return parsed_ds

def _parse_image_function(example_proto, image_description_dict):
    # Parse the input tf.Example proto using the dictionary above.
    return tf.io.parse_single_example(example_proto, image_description_dict)

def tf_record_builder(data_dir, tf_records_dir, tf_filename,
                      vis=False, uv=False,
                      applier=None, label_model=None):
    if not os.path.isdir(tf_records_dir):
        print("Creating tf records dir %s" % tf_records_dir)
        os.makedirs(tf_records_dir)

    list_ds = tf.data.Dataset.list_files(data_dir+"/*",
                                         shuffle=True)
    num_samples = tf.data.experimental.cardinality(list_ds).numpy()
    num_batches = np.ceil(num_samples / IMAGES_PER_TFR).astype(np.int)
    
    record_number = 0
    for batch in tqdm(list_ds.batch(IMAGES_PER_TFR), total=num_batches):
        tf_records_path = os.path.join(tf_records_dir, tf_filename+"_{:03d}.tfr".format(record_number))
        record_number += 1

        batch_ds = tf.map_fn(lambda x:  _serialise_image(x, vis=vis,
                                                            uv=uv,
                                                            image_shape=[IMG_WIDTH, IMG_HEIGHT],
                                                            applier=applier,
                                                            label_model=label_model),
                             batch)
        with tf.io.TFRecordWriter(tf_records_path) as writer:
            for tf_example in batch_ds:
                writer.write(tf_example.numpy())

def dir_dataloader(data_dir, shuffle=True,
                   vis=False, uv=False):
    if not (vis and uv):
        print("Error must have at leats one of vis and uv")
        
    list_ds = tf.data.Dataset.list_files(data_dir+"/*",
                                         shuffle=shuffle)
    print("Found %i images" %  tf.data.experimental.cardinality(list_ds).numpy())

    ds = list_ds.map(lambda x: _process_image(x, vis=vis,
                                                 uv=uv,
                                                 image_shape=[IMG_WIDTH, IMG_HEIGHT])
                    )

    return ds, list_ds

def _process_image(file_path,
                   vis=False, uv=False,
                   image_shape=[IMG_WIDTH, IMG_HEIGHT]):
    split_file_path = tf.strings.split(file_path, os.path.sep)
    base_dir = tf.strings.reduce_join(split_file_path[:-2],
                                      separator=os.path.sep)
    
    filename = split_file_path[-1]
    split_filename = tf.strings.split(filename, "_")
    base_name = tf.strings.reduce_join(split_filename[:-5],
                                       separator="_")
        
    output_list = []

    if vis:
        vis_filename = base_name + "_00_99_002_001_RAI.jpg"
        vis_path = tf.strings.join([base_dir, VIS_DIR, vis_filename],
                                    os.path.sep)
        vis_img = _decode_img(vis_path, image_shape)
            
        output_list.append(vis_img)
    if uv:
        uv_filename = base_name + "_U0_99_002_002_RAI.jpg"
        uv_path = tf.strings.join([base_dir, UV_DIR, uv_filename],
                                  os.path.sep)
        uv_img = _decode_img(uv_path, image_shape)

        output_list.append(uv_img)
    
    return tuple(output_list)

def _serialise_image(file_path,
                     vis=False, uv=False,
                     image_shape=[IMG_WIDTH, IMG_HEIGHT],
                     applier=None,
                     label_model=None):
    data_tuple = _process_image(file_path,
                                vis=vis, uv=uv,
                                image_shape=image_shape)

    data_dict = {"vis": None,
                 "uv": None, 
                 "soft_label": None}

    if vis and uv: 
        data_dict["vis"] = data_tuple[0]
        data_dict["uv"] = data_tuple[1]
    elif uv:
        data_dict["vis"] = data_tuple[0]
    elif uv:
        data_dict["uv"] = data_tuple[0]

    L_sample = applier.apply([data_tuple],
                             progress_bar=False)
    data_dict["soft_label"] = label_model.predict_proba(L_sample)
    
    return _serialize_example(**data_dict)

def _decode_img(file_path,
                image_shape,
                num_channels=3):
    img = tf.io.read_file(file_path)
        
    img = tf.io.decode_image(img, 
                             channels=num_channels, 
                             dtype=tf.float32, 
                             expand_animations=False)

    return tf.image.resize(img, image_shape)

def _serialize_example(vis=None, uv=None, soft_label=None):
    """
    Creates a tf.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {}
    if vis is not None:
        feature["vis"] = _bytes_feature(tf.io.serialize_tensor(vis))
    if uv is not None:
        feature["uv"] = _bytes_feature(tf.io.serialize_tensor(uv)) 
    if soft_label is not None:
        feature["soft_label"] = _bytes_feature(tf.io.serialize_tensor(soft_label[0]))
    
    # Create a Features message using tf.train.Example.

    image_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  
    return image_proto.SerializeToString()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def test_builder():
    data_dir = "/home/chris/Documents/C3/data/train/visible"
    tf_records_dir = "/home/chris/Documents/C3/data/tf_records/train"
    tf_filename = "c3_uv_train"

    tf_record_builder(data_dir, tf_records_dir, tf_filename, vis=True, uv=True)

    tfr_ds = tf_record_reader(tf_records_dir, vis=True, uv=True)

    train_ds = tfr_ds.map(lambda x: process_input(x, vis=True, uv=True), 
                          num_parallel_calls=AUTOTUNE)
    train_ds = prepare_for_training(train_ds,
                                    cache="test.cache",
                                    batch_size=BATCH_SIZE)

if __name__ == "__main__":
    test_builder()