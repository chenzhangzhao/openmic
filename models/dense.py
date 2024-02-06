import tensorflow as tf
from keras import regularizers
from keras.layers import Dense, Dropout, Flatten, Input


class DenseClassifier:
    def __init__(
        self,
        input_shape: tuple,
        num_classes: int,
        dropout: float = 0.5,
        l2_regularization: float = None,
    ):
        self.num_classes = num_classes
        self.input_shape = input_shape
        # print(input_shape)
        # exit(0)
        self.dropout = dropout
        self.l2_regularization = l2_regularization
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential(
            [
                Input(shape=self.input_shape, dtype=tf.float32),
                Flatten(),
                Dense(
                    1024,
                    activation="relu",
                    kernel_regularizer=regularizers.L2(l2=self.l2_regularization),
                )
                if self.l2_regularization
                else Dense(1024, activation="relu"),
                Dropout(self.dropout),
                Dense(self.num_classes, activation="softmax"),
            ]
        )

        return model

    def forward(self, X):
        return self.model(X)
