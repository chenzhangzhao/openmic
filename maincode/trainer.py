import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping
from maincode.function import timer, make_dir, create_uid, export_json


class MetricsHistory(tf.keras.callbacks.Callback):
    """Custom callback to track metrics during training."""

    def __init__(self):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        for metric, value in logs.items():
            if metric not in self.history:
                self.history[metric] = []
            # if not isinstance(value,float):
            #     self.history[metric].append(max(value))
            # else:
            #     self.history[metric].append(value)
            self.history[metric].append(value)

    def save_to_file(self, filename):
        export_json(self.history, filename)


class ModelTrainer:
    def __init__(
        self, model, learning_rate, loss="categorical_crossentropy", export_path=None
    ):
        """Initialize the model trainer."""
        self.runid = create_uid()
        self.model = model
        self.export_path = f"{export_path}\\{self.runid}"
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.loss = loss
        self.metrics = [keras.metrics.Precision(),keras.metrics.Recall(),keras.metrics.F1Score(average='micro')]
        self.metrics_history = MetricsHistory()
        make_dir(self.export_path)  # create path for results

    def save_model(self):
        print(f"==> Saving model to {self.export_path} ...")
        self.model.model.save(f"{self.export_path}\\model.h5")

    def compile(self):
        self.model.model.compile(
            optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )

    @timer
    def train(self, train_dataset, epochs=10, validation_data=None):
        history = self.model.model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[
                self.metrics_history,
                EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
            ],
            validation_data=validation_data,
        )
        self.metrics_history.save_to_file(f"{self.export_path}\\training.json")
        return history

    def evaluate(self, test_dataset):
        evaluation = self.model.model.evaluate(test_dataset)
        evaluation_dict = dict(zip(self.model.model.metrics_names, evaluation))
        export_json(evaluation_dict, f"{self.export_path}\\evaluation.json")
