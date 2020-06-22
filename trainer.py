from __future__ import division

import argparse
import os
import sys
import datetime
import logging
import time
import cv2
import torch
import h5py
import json

import deepcrystalModel

from tqdm import tqdm
from glob import glob
# from sqlalchemy import create_engine
# from keras.preprocessing.image import ImageDataGenerator
# from keras.utils import HDF5Matrix
from snorkel.labeling import labeling_function, LFApplier, LFAnalysis
from snorkel.labeling.model import LabelModel
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, concatenate, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.abspath('../hopless'))
from model import DciModel

from data import dir_dataloader, tf_record_builder, tf_record_reader, process_input, prepare_for_training

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

print(torch.__version__)

LOGGER = logging.getLogger("c3.uv_trainer")
DEBUG = False

LOCAL_BASE_DIR = "/home/chris/Documents/C3/data"
CACHE_DIR = "/media/chris/SharedStorage/CSIRO/c3/tf_cache"
if not os.path.isdir(CACHE_DIR):
    LOGGER.info("Creating cache dir %s" % CACHE_DIR)
    os.makedirs(CACHE_DIR)
TFR_DIR = "/media/chris/SharedStorage/CSIRO/c3/tfr"
if not os.path.isdir(TFR_DIR):
    LOGGER.info("Creating cache dir %s" % TFR_DIR)
    os.makedirs(TFR_DIR)
AUTOTUNE = tf.data.experimental.AUTOTUNE

CLEAR = 0
PRECIPITATE = 1
CRYSTAL = 2
OTHER = 3
ABSTAIN = -1
CLASS_NAMES = [0, 1, 2, 3]

SEED = 42
SNORKEL_BATCH_SIZE = 1
SOFT_LABEL_BATCH_SIZE = 512
TRAINING_BATCH_SIZE = 16

TARGET_SIZE = (224, 224)

MARCO_MODEL_PATH = "/home/chris/Documents/C3/data/MARCO"
MARCO_CLASSES = {"Clear": CLEAR,
                 "Precipitate": PRECIPITATE,
                 "Crystals": CRYSTAL,
                 "Other": OTHER}

# DEEPCRYSTAL_WEIGHTS_FILE = "deepcrystal.t7"
DEEPCRYSTAL_WEIGHTS_FILE = "../hopless/di_densenet121.ckpt"
DEEPCRYSTAL_MODEL_FILE = "deepcrystalModel.pth"
HEADERS = ['Alternate_Spectrum_Negative', 'Alternate_Spectrum_Positive',
           'Clear', 'Contaminants', 'Dry', 'Failure', 'Macro', 'Micro', 'Phase Separation',
           'Precipitate Amorphous', 'Precipitate Non-Amorphous', 'Skin', 'Spherulite']

L_TRAIN_FILE = "L_train.npy"
L_TEST_FILE = "L_test.npy"
# L_TRAIN_FILE = "L_train_small.npy"
# L_TEST_FILE = "L_test_small.npy"
SCORE_FILE = "/home/chris/Documents/C3/data/uvscores.json"

def main(args):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    begin_time = time.time()
    LOGGER.info("Processing began at %s", time.ctime(begin_time))
    LOGGER.info(" ")

    LOGGER.info("Preparing training generator...")
    train_data_dir = os.path.join(LOCAL_BASE_DIR, "train_mid", "visible")
    train_cache_filename = os.path.join(CACHE_DIR, "train.cache")
    train_ds, _ = dir_dataloader(train_data_dir,
                                 vis=True, uv=True)
    # train_ds = prepare_for_training(train_ds,
    #                                 cache=train_cache_filename,
    #                                 batch_size=SNORKEL_BATCH_SIZE)
    LOGGER.info("... done")
    LOGGER.info("Preparing test generator...")
    test_data_dir = os.path.join(LOCAL_BASE_DIR, "test_full", "visible")
    test_ds, filename_ds = dir_dataloader(test_data_dir, shuffle=False,
                                          vis=True, uv=True)
    LOGGER.info("... done")
    LOGGER.info(" ")

    LOGGER.info("Getting test labels...")
    with open(SCORE_FILE) as score_json:
        test_labels = json.load(score_json)
    score_df = pd.DataFrame(list(test_labels.items())[0][1])
    score_df["basename"] = np.array(["_".join(imagename.split("_")[:-5]) for imagename in score_df["IMAGENAME"].values])
    test_labels_ds = filename_ds.map(lambda x: get_scores(x, score_df),
                                     num_parallel_calls=AUTOTUNE)
    label_array = np.array(list(tfds.as_numpy(test_labels_ds)))
    labels_ds = filename_ds.map(lambda x: get_scores(x, score_df, one_hot=True),
                                num_parallel_calls=AUTOTUNE)
    labelled_test_ds = tf.data.Dataset.zip((test_ds, labels_ds))

    test_cache_filename = os.path.join(CACHE_DIR, "test.cache")
    test_ds = prepare_for_training(test_ds,
                                   cache=test_cache_filename,
                                   batch_size=SNORKEL_BATCH_SIZE,
                                   shuffle_buffer_size=None)
    test_cache_filename = os.path.join(CACHE_DIR, "labelled_test.cache")
    labelled_test_ds = prepare_for_training(labelled_test_ds,
                                            cache=test_cache_filename,
                                            batch_size=TRAINING_BATCH_SIZE,
                                            shuffle_buffer_size=None)
    LOGGER.info("... done")
    LOGGER.info(" ")

    LOGGER.info("Creating multi-image network...")
    base_model = ResNet50V2(include_top=False,
                            weights="imagenet",
                            input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    # base_model = MobileNetV2(include_top=False,
    #                          weights="imagenet",
    #                          alpha=1.0,
    #                          input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    global_average_layer = GlobalAveragePooling2D()
    feature_extractor = tf.keras.Sequential([base_model,
                                             global_average_layer,
                                            ])
    vis_input = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    vis_features = feature_extractor(vis_input)

    uv_input = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    uv_features = feature_extractor(uv_input)

    combined = concatenate([vis_features, uv_features])

    z = Dense(256, activation="relu")(combined)
    z = Dense(128, activation="relu")(z)
    z = Dense(4, activation="softmax")(z)
    night_vision = Model(inputs=[vis_input, uv_input],
                         outputs=z)
    opt = Adam(lr=1e-4, decay=1e-3 / 200)
    night_vision.compile(loss="categorical_crossentropy", optimizer=opt,
                         metrics=["accuracy"])
    night_vision.summary()
    LOGGER.info("... done")
    LOGGER.info(" ")

    LOGGER.info("Loading MARCO model...")
    predicter = tf.saved_model.load(MARCO_MODEL_PATH).signatures["predict_images"]
    LOGGER.info("... done")
    # Category-based LFs
    @labeling_function()
    def lf_marco(x):
        vis, _ = x

        str_encode = tf.io.encode_jpeg(tf.cast(tf.squeeze(vis)*255, tf.uint8))
        results = predicter(tf.convert_to_tensor([str_encode]))

        return MARCO_CLASSES[results["classes"][0][0].numpy().decode()]

    LOGGER.info("Loading DeepCrystal model...")
    model = deepcrystalModel.deepcrystalModel
    model.load_state_dict(torch.load(DEEPCRYSTAL_MODEL_FILE))
    model.cuda()
    model.eval()
    LOGGER.info("... done")

    # LOGGER.info("Loading DeepCrystal model...")
    # model = DciModel()
    # model.load_weights(DEEPCRYSTAL_WEIGHTS_FILE)
    # LOGGER.info("... done")

    # image = tf.image.decode_jpeg(jpeg_bytes)
    # min_dim = image.shape[0] if image.shape[0] < image.shape[1] else image.shape[1]
    # min_dim = tf.cast(min_dim, tf.int32)
    # image = tf.image.resize_with_crop_or_pad(image, min_dim, min_dim)
    # image = tf.image.resize(image, (224,224))
    # image /= 255.

    @labeling_function()
    def lf_deepcrystal_uv(x):
        _, uv = x
        uv = (np.squeeze(uv.numpy())*255).astype(np.uint8)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]
                                        )
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize
                                        ])
        img_tensor = preprocess(uv)
        img_tensor.unsqueeze_(0)

        batch_variable = torch.autograd.Variable(img_tensor)
        logits = model.forward(batch_variable.cuda())
        softie = torch.nn.Softmax(dim=1)

        local_softmax = softie(logits.cuda()).cpu().data.numpy()
        if np.argmax(local_softmax[0])==1:
            return CRYSTAL
        else:
            return ABSTAIN

    @labeling_function()
    def lf_deepcrystal_vis(x):
        vis, _ = x
        vis = (np.squeeze(vis.numpy())*255).astype(np.uint8)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]
                                        )
        preprocess = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         normalize
                                        ])
        img_tensor = preprocess(vis)
        img_tensor.unsqueeze_(0)

        batch_variable = torch.autograd.Variable(img_tensor)
        logits = model.forward(batch_variable.cuda())
        softie = torch.nn.Softmax(dim=1)

        local_softmax = softie(logits.cuda()).cpu().data.numpy()
        if np.argmax(local_softmax[0])==2:
            return CLEAR
        elif np.argmax(local_softmax[0])==6 or np.argmax(local_softmax[0])==7:
            return CRYSTAL
        elif np.argmax(local_softmax[0])==9 or np.argmax(local_softmax[0])==10:
            return PRECIPITATE
        else:
            return OTHER

    LOGGER.info("Applying weak labels...")
    lfs = [lf_marco, lf_deepcrystal_uv, lf_deepcrystal_vis]
    applier = LFApplier(lfs)
    if os.path.isfile(L_TRAIN_FILE):
        LOGGER.info("... loading L_train from file")
        L_train = np.load(L_TRAIN_FILE)
    else:
        num_train_images = tf.data.experimental.cardinality(train_ds).numpy()
        print(num_train_images)
        L_train_mm = np.memmap(os.path.join(CACHE_DIR, "numpy.cache"),
                               dtype="float32",
                               mode="w+",
                               shape=(num_train_images, len(lfs)))
        snorkel_cache_filename = os.path.join(CACHE_DIR, "snorkel.cache")
        for b, batch in enumerate(train_ds.batch(SOFT_LABEL_BATCH_SIZE)):
            b_ds = tf.data.Dataset.from_tensor_slices(batch)
            b_ds = prepare_for_training(b_ds,
                                        cache=True,
                                        batch_size=SNORKEL_BATCH_SIZE)
            L_train = applier.apply(b_ds)
            L_train_mm[b*SOFT_LABEL_BATCH_SIZE:b*SOFT_LABEL_BATCH_SIZE+len(L_train), :] = L_train[:, :]
            # os.remove(snorkel_cache_filename)
        LOGGER.info("... saving L_train to file")
        # np.save(L_TRAIN_FILE, L_train)
    if os.path.isfile(L_TEST_FILE):
        LOGGER.info("... loading L_test from file")
        L_test = np.load(L_TEST_FILE)
    else:
        L_test = applier.apply(test_ds)
        LOGGER.info("... saving L_test to file")
        # np.save(L_TEST_FILE, L_test)
    LOGGER.info("... done")

    print(LFAnalysis(L_test, lfs).lf_summary(label_array))
    label_model = LabelModel(cardinality=4, verbose=True)
    label_model.fit(L_train_mm, seed=SEED, lr=0.001, log_freq=100, n_epochs=2000)
    del L_train_mm

    print(label_model.score(L_test, label_array, metrics=["f1_micro", "f1_macro", "accuracy"]))

    predicted_labels = label_model.predict(L_test)

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))

    print("Snorkel")
    plot_confusion_matrix(label_array,
                          predicted_labels,
                          ("Abstain", "Clear", "Precipitate", "Crystal", "Other"),
                          normalize=True,
                          title="Snorkel",
                          ax=ax[0, 0])

    print("MARCO")
    plot_confusion_matrix(label_array,
                          L_test[:, 0],
                          ("Abstain", "Clear", "Precipitate", "Crystal", "Other"),
                          normalize=True,
                          title="MARCO",
                          ax=ax[0, 1])

    print("DCUV")
    plot_confusion_matrix(label_array,
                          L_test[:, 1],
                          ("Abstain", "Clear", "Precipitate", "Crystal", "Other"),
                          normalize=True,
                          title="DeepCrystal (UV)",
                          ax=ax[1, 0])

    print("DCRGB")
    plot_confusion_matrix(label_array,
                          L_test[:, 2],
                          ("Abstain", "Clear", "Precipitate", "Crystal", "Other"),
                          normalize=True,
                          title="DeepCrystal (Vis)",
                          ax=ax[1, 1])

    fig.savefig("confusion_matrices.png")

    print("Building dataset...")
    tf_records_dir = os.path.join(TFR_DIR, "tf_records", "train")
    if not os.path.isdir(tf_records_dir):
        LOGGER.info("Creating tfr dir %s" % tf_records_dir)
        os.makedirs(tf_records_dir)
    tf_filename = "c3_uv_train"
    tf_record_builder(train_data_dir, tf_records_dir, tf_filename,
                      vis=True, uv=True,
                      applier=applier, label_model=label_model)

    train_cache_filename = os.path.join(CACHE_DIR, "labelled_train.cache")
    labelled_ds = tf_record_reader(tf_records_dir, vis=True, uv=True)
    labelled_ds = labelled_ds.map(lambda x: process_input(x, vis=True, uv=True, image_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3)),
                                  num_parallel_calls=AUTOTUNE)
    labelled_ds = prepare_for_training(labelled_ds,
                                       cache=train_cache_filename,
                                       batch_size=TRAINING_BATCH_SIZE)

    print("Training CNN...")

    history = night_vision.fit(labelled_ds,
                               validation_data=labelled_test_ds,
                               callbacks=[],
                               epochs=50,
                               verbose=1,
                               shuffle="batch"
                              )

    plt.figure()
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.legend(["train", "val"])
    plt.savefig("loss_plots.png")

    print("Testing night vision...")
    night_vision_pred_list = []
    for batch in test_ds:
        night_vision_pred_list.append(night_vision.predict(batch)[0])
    night_vision_pred_array = np.array(night_vision_pred_list)
    night_vision_preds = np.argmax(night_vision_pred_array, 1)
    print("Night vision test accuracy: %f" % accuracy_score(label_array, night_vision_preds))

    print("Plotting night vision confusion matrix...")
    ax = plt.figure()
    plot_confusion_matrix(label_array,
                          night_vision_preds,
                          ("Abstain", "Clear", "Precipitate", "Crystal", "Other"),
                          normalize=True,
                          title="Night Vision")
    plt.savefig("night_vision_confusion_matrix.png")
    print("... done")

    print("Testing MARCO...")
    marco_pred_list = []
    for batch in test_ds:
        vis, _ = batch
        str_encode = tf.io.encode_jpeg(tf.cast(tf.squeeze(vis)*255, tf.uint8))
        results = predicter(tf.convert_to_tensor([str_encode]))
        marco_pred_list.append(MARCO_CLASSES[results["classes"][0][0].numpy().decode()])
    marco_pred_array = np.array(marco_pred_list)
    print("MARCO test accuracy: %f" % accuracy_score(label_array, marco_pred_array))

# def transfer_to_local(source_path,
#                       dest_dir):
#     start_time = time.time()
#     LOGGER.info("... copying visible images")
#     dest_dir = os.path.join(os.environ["TMPDIR"], dest_dir)
#     rsync_string = "rsync -azv --progress {} {}".format(source_path, dest_dir)
#     LOGGER.info("... executing rsync: %s", rsync_string)
#     os.system(rsync_string)
#     LOGGER.info("... done in %fs", time.time() - start_time)

#     final_dir = source_path.split("/")[-1]

#     return os.path.join(dest_dir, final_dir)

def get_scores(image_path,
               scores_df,
               one_hot=False):
    names_t = tf.convert_to_tensor(scores_df["basename"].values)
    scores_t = tf.convert_to_tensor(scores_df["SCORE"].values)
    split_file_path = tf.strings.split(image_path, os.path.sep)
    imagename = tf.strings.split(split_file_path[-1], ".")[0]
    split_filename = tf.strings.split(imagename, "_")
    base_name = tf.strings.reduce_join(split_filename[:-5],
                                       separator="_")
    scores = tf.boolean_mask(scores_t,
                             tf.math.equal(names_t, base_name))

    image_label = ABSTAIN
    if scores[-1]=="8":
        image_label = CLEAR
    elif scores[-1]=="18":
        image_label = PRECIPITATE
    elif scores[-1]=="28":
        image_label = CRYSTAL
    elif scores[-1]=="38":
        image_label = OTHER
    elif scores[-1]=="58":
        image_label = ABSTAIN

    if one_hot:
        image_label = image_label == CLASS_NAMES

    return image_label

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          ax=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    return ax

def c3_uv_trainer_cli():
    # Import config values
    parser = argparse.ArgumentParser(description="""Train
                                     a UV classifier with Snorkel.""")

    parser.add_argument("--data_dir",
                        default=LOCAL_BASE_DIR,
                        type=str,
                        help="Directory to copy images to.")

    args = parser.parse_args()

    LOGGER.info(" ")
    LOGGER.info("//////////////////////////////////////////")
    LOGGER.info("//                                      //")
    LOGGER.info("//      C3 UV IMAGE TRAINER             //")
    LOGGER.info("//                                      //")
    LOGGER.info("//////////////////////////////////////////")
    LOGGER.info(" ")

    if not os.path.isdir(args.data_dir):
        LOGGER.info("Creating data dir - %s", args.data_dir)
        os.makedirs(args.data_dir)

    main(args)

def set_up_logger(log_dir="./logs"):
    """This function initialises the logger.
    We set up a logger that prints both to the console at the information level
    and to file at the debug level. It will store in the /temp directory on
    *NIX machines and in the local directory on windows.
    """
    timestamp = datetime.datetime.now()

    logfile_name = 'c3_uv_trainer-{0:04}-{1:02}-{2:02}-{3:02}{4:02}{5:02}.log'\
                   .format(timestamp.year,
                           timestamp.month,
                           timestamp.day,
                           timestamp.hour,
                           timestamp.minute,
                           timestamp.second)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    logfile_name = os.path.join(log_dir,
                                logfile_name)

    logging.basicConfig(filename=logfile_name,
                        level=logging.DEBUG)

    console_logger = logging.StreamHandler()
    if DEBUG:
        console_logger.setLevel(logging.DEBUG)
    else:
        console_logger.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)-9s: %(levelname)-8s %(message)s')
    console_logger.setFormatter(console_formatter)
    logging.getLogger('').addHandler(console_logger)

    LOGGER.info('All logging will be written to %s', logfile_name)

if __name__ == '__main__':
    set_up_logger()
    c3_uv_trainer_cli()
