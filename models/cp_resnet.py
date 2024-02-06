import tensorflow as tf
from keras import layers

def initialize_weights(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        tf.keras.initializers.he_normal()(layer.weights[0])
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        tf.keras.initializers.Ones()(layer.weights[0])
        tf.keras.initializers.Zeros()(layer.weights[1])
    elif isinstance(layer, tf.keras.layers.Dense):
        tf.keras.initializers.Zeros()(layer.weights[1])

class BasicBlock(tf.keras.layers.Layer):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride, k1=3, k2=3):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(
            out_channels,
            kernel_size=k1,
            strides=stride,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            out_channels,
            kernel_size=k2,
            strides=1,
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()

        self.shortcut = tf.keras.Sequential()
        if in_channels != out_channels:
            self.shortcut.add(
                layers.Conv2D(
                    out_channels,
                    kernel_size=1,
                    strides=stride,
                    padding='valid',
                    use_bias=False,
                    kernel_initializer='he_normal'))
            self.shortcut.add(layers.BatchNormalization())

    def call(self, x):
        y = tf.nn.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y += self.shortcut(x)
        y = tf.nn.relu(y)
        return y

class CPResNet(tf.keras.Model):
    def __init__(self, rho, input_shape, num_classes=20,dropout: float = 0.5,l2_regularization: float = None,base_channels=128):
        super(CPResNet, self).__init__()

        self.in_c = tf.keras.Sequential([
            layers.Conv2D(base_channels, 5, strides=2, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU()
        ])

        extra_kernel_rf = rho - 7

        self.stage1 = self._make_stage(
            base_channels, base_channels, 4,
            maxpool={1: 2, 2: 2, 4: 2},
            k1s=(
                3,
                3 - (-extra_kernel_rf > 6) * 2,
                3 - (-extra_kernel_rf > 4) * 2,
                3 - (-extra_kernel_rf > 2) * 2),
            k2s=(
                1,
                3 - (-extra_kernel_rf > 5) * 2,
                3 - (-extra_kernel_rf > 3) * 2,
                3 - (-extra_kernel_rf > 1) * 2))

        self.stage2 = self._make_stage(
            base_channels, base_channels * 2, 4,
            k1s=(
                3 - (-extra_kernel_rf > 0) * 2,
                1 + (extra_kernel_rf > 1) * 2,
                1 + (extra_kernel_rf > 3) * 2,
                1 + (extra_kernel_rf > 5) * 2),
            k2s=(1,
                3 - (-extra_kernel_rf > 5) * 2,
                3 - (-extra_kernel_rf > 3) * 2,
                3 - (-extra_kernel_rf > 1) * 2))

        self.stage3 = self._make_stage(
            base_channels * 2, base_channels * 4, 4,
            k1s=(
                1 + (extra_kernel_rf > 7) * 2,
                1 + (extra_kernel_rf > 9) * 2,
                1 + (extra_kernel_rf > 11) * 2,
                1 + (extra_kernel_rf > 13) * 2),
            k2s=(
                1 + (extra_kernel_rf > 8) * 2,
                1 + (extra_kernel_rf > 10) * 2,
                1 + (extra_kernel_rf > 12) * 2,
                1 + (extra_kernel_rf > 14) * 2))

        self.feed_forward = layers.Dense(num_classes, kernel_initializer='zeros')

    def _make_stage(self, in_channels, out_channels, n_blocks, maxpool=set(), k1s=[3, 3, 3, 3, 3, 3],
                    k2s=[3, 3, 3, 3, 3, 3]):
        stage = tf.keras.Sequential()

        for index in range(n_blocks):
            stage.add(BasicBlock(in_channels, out_channels, stride=1, k1=k1s[index], k2=k2s[index]))

            in_channels = out_channels
            if index + 1 in maxpool:
                stage.add(layers.MaxPooling2D(pool_size=maxpool[index + 1]))

        return stage

    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        x = self.in_c(x)
        output_1 = self.stage1(x)
        output_2 = self.stage2(output_1)
        output_3 = self.stage3(output_2)
        output = tf.reduce_mean(output_3, axis=[1, 2])
        output = self.feed_forward(output)
        return dict(
            logits=tf.nn.sigmoid(output),
            scores=output,
            output_1=tf.reduce_mean(output_1, axis=-2, keepdims=False),
            output_2=tf.reduce_mean(output_2, axis=-2, keepdims=False),
            output_3=tf.reduce_mean(output_3, axis=-2, keepdims=False)
        )
