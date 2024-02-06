import tensorflow as tf
import config as config


def _parse_function(proto, num_classes):
    """Parse a single tfrecord proto."""
    feature_description = {
        "feature": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64),
    }
    parsed_features = tf.io.parse_single_example(proto, feature_description)
    # decode the feature
    feature = tf.io.parse_tensor(parsed_features["feature"], out_type=tf.float32)
    label = tf.one_hot(
        parsed_features["label"], depth=num_classes
    )  # encode label as one-hot vector
    return feature, label


class AudioDataLoader:
    """Data loader for the audio dataset."""

    def __init__(self, file_list, batch_size=32, dataset=None, shuffle=True):
        if dataset in [
            "validation",
            "test",
        ]:  # make sure to use non-augmented data for validation and test
            self.file_list = [
                f"{config.TARGET_AUDIO_DIR.replace('augmented', 'processed')}\\{config.MODEL_NAME}_features\\{fp}.tfrecord"
                for fp in file_list
            ]
        elif dataset == "train":
            self.file_list = [
                f"{config.TARGET_AUDIO_DIR}\\{config.MODEL_NAME}_features\\{fp}.tfrecord"
                for fp in file_list
            ]
            # print(self.file_list[0])
            # exit(0)
        else:
            raise ValueError(
                f"Dataset must be one of 'train', 'validation', or 'test', got {dataset}"
            )

        self.num_classes = config.N_CLASSES
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return len(self.file_list)

    def create_dataset(self):
        """Create a tf.data.Dataset object."""
        # print(self.file_list[0])
        # exit(0)
        dataset = tf.data.TFRecordDataset(self.file_list)
        dataset = dataset.map(lambda x: _parse_function(x, self.num_classes))
        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=len(self.file_list))

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        return dataset
