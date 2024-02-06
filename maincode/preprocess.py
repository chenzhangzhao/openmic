# -*- coding: utf-8 -*-
"""preprocess.py"""
from maincode.function import load_json
import glob
import os
import librosa
import numpy as np
import tensorflow_hub as hub
import config as config
import pandas as pd
import tensorflow as tf
import librosa
import random
import numpy as np


class Augmentations:
    """Audio augmentations"""

    def __init__(self):
        pass

    def time_stretch(self, audio, rate=None):
        if rate is None:
            rate = random.uniform(0.8, 1.2)  # random rate between 0.8x and 1.2x
        return librosa.effects.time_stretch(y=audio, rate=rate)

    def pitch_shift(self, audio, sr, n_steps=None):
        if n_steps is None:
            n_steps = random.uniform(-1, 1)  # random pitch shift between -1 and 1
        return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

    def add_noise(self, audio, noise_level=None):
        if noise_level is None:
            noise_level = random.uniform(0.001, 0.005)  # add random noise
        noise = np.random.randn(len(audio))
        return audio + noise_level * noise

    def random_crop(self, audio, sr, target_duration):
        target_length = int(sr * target_duration)
        if len(audio) >= target_length:
            start = random.randint(0, len(audio) - target_length)
            return audio[start : start + target_length]
        return audio

    def change_volume(self, audio, volume_change=None):
        if volume_change is None:
            volume_change = random.uniform(0.5, 1.5)  # random volume change
        return audio * volume_change

    def pad_clip(self, audio, sr, target_duration):
        target_length = int(sr * target_duration)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        elif len(audio) > target_length:
            audio = audio[:target_length]
        return audio

    def apply_augmentations(self, audio, sr, target_duration):
        audio = self.time_stretch(audio)
        audio = self.pitch_shift(audio, sr)
        audio = self.add_noise(audio)
        audio = self.change_volume(audio)
        audio = self.random_crop(audio, sr, target_duration)
        audio = self.pad_clip(audio, sr, target_duration)
        return audio

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize(embedding, label):
    """Serialize the embeddings and label to a tfrecord proto."""
    feature = {
        "feature": _bytes_feature(tf.io.serialize_tensor(embedding)),
        "label": _int64_feature(label),
    }
    proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return proto.SerializeToString()


def write_tfrecord(target_path, embeddings, label):
    """Write the embeddings and label to a tfrecord file."""
    # wintarget_path=target_path.split('\\')
    # tfwintarget_path=os.path.join(wintarget_path[0],wintarget_path[1])
    # if not os.path.exists(tfwintarget_path):
    #     os.makedirs(tfwintarget_path)
    with tf.io.TFRecordWriter(target_path) as writer:
        serialized_example = serialize(embeddings.numpy(), label)
        writer.write(serialized_example)


def consolidate_labels(labels_path):
    """Consolidate the labels to the highest relevance instrument per class."""
    agg_labels = pd.read_csv(labels_path)
    agg_labels = (
        agg_labels.groupby("sample_key")
        .agg({"instrument": lambda x: x.iloc[np.argmax(x.values)]})
        .reset_index()
    )
    return agg_labels


def load_label(labels_df, class_map, file_name):
    """Load the label for a given file name."""
    file_name = file_name.split("\\")[-1].split(".")[0]
    label = labels_df.loc[labels_df["sample_key"] == file_name, "instrument"].values[0]
    label_index = class_map[label]
    return label_index


def pad_clip(audio, sr, target_duration):
    target_length = int(sr * target_duration)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    elif len(audio) > target_length:
        audio = audio[:target_length]
    return audio


def load_and_preprocess_audio(
    model, model_name, out_dim, file_path, target_duration=10, target_sr=16000
):
    """Load, resample and feature extract from the audio files."""
    audio, sr = librosa.load(file_path, sr=None, mono=True)  # load audio

    if sr != target_sr:  # resample as needed
        audio = librosa.resample(y=audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    if config.AUGMENT:
        audio = augmentations.apply_augmentations(audio, sr, target_duration)

    else:
        audio = pad_clip(audio, sr, target_duration)

    # forward pass to get the embeddings
    output = model(audio)

    if model_name == "vggish":
        embeddings = output
    elif model_name == "yamnet":
        _, embeddings, log_mel_spectrogram = output

    else:
        raise ValueError("Invalid model name")

    # sanity check
    embeddings.shape.assert_is_compatible_with([None, out_dim])

    return embeddings


if __name__ == "__main__":
    for model_name, model_args in config.MODELS.items():
        model_url = model_args.get("url")
        out_dim = model_args.get("out_dim")
        model = hub.load(model_url)

        filepaths = glob.glob(f"{config.AUDIO_DIR}\\*\\*.ogg")
        target_filepaths = ["\\".join(fp.split("\\")[-1:]) for fp in filepaths]
        fpath=os.path.join(config.TARGET_AUDIO_DIR,model_name+"_features")
        if not os.path.exists(fpath):
            os.mkdir(fpath)
        labels_df = consolidate_labels(config.LABELS_PATH)
        class_map = load_json(config.LABELS_MAP_PATH)
        #  augmentations
        augmentations = Augmentations()
        # preprocess each file and save to thrcord
        for i in range(len(filepaths)):
            fp=filepaths[i]
            target_path = f"{config.TARGET_AUDIO_DIR}\\{model_name}_features\\{target_filepaths[i]}".replace(".ogg", ".tfrecord")
            embeddings = load_and_preprocess_audio(model,model_name,out_dim,fp,
            target_duration=config.TARGET_DURATION,target_sr=config.TARGET_SAMPLE_RATE,)
            label = load_label(labels_df=labels_df,class_map=class_map,file_name=target_filepaths[i],)
            write_tfrecord(target_path, embeddings, label)
