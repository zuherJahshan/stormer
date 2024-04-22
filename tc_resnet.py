import tensorflow as tf

class TemporalBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int,
        strides: int = 1,
        dilation_rate: int = 1,
        activation: str = None,
        kernel_regularizer = None,
        dropout: float = 0.0,
    ):
        super(TemporalBlock, self).__init__()
        # Primitive properties
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilation_rate = dilation_rate

        self.conv = tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=[kernel_size, 1],
            strides=[strides, 1],
            padding="same",
            dilation_rate=[dilation_rate, 1],
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
        
        self.dropout = tf.keras.layers.Dropout(dropout)

    
    def call(self, inputs):
        # We assume input is of the shape [batch, time, 1, freq]
        out_timesteps = tf.cast(tf.shape(inputs)[1] / self.strides, tf.int32)
        padding = (self.kernel_size - 1) * self.dilation_rate
        if padding > 0:
            inputs = tf.pad(inputs, tf.constant([(0, 0,), (padding, 0), (0, 0), (0, 0)]) * padding)
        outputs = self.conv(inputs)
        return self.dropout(outputs)[:, :out_timesteps, :, :]
    

class TempResBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        type: int, # either 1 or 2
        filters: int,
        kernel_size: int,
        kernel_regularizer = None,
        dropout: float = 0.0,
    ):
        super(TempResBlock, self).__init__()
        # Primitive properties
        self.type = type

        self.tb1 = TemporalBlock(
            filters=filters,
            kernel_size=kernel_size,
            strides=type,
            kernel_regularizer=kernel_regularizer,
            dropout=dropout,
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation("relu")
        self.tb2 = TemporalBlock(
            filters=filters,
            kernel_size=kernel_size,
            kernel_regularizer=kernel_regularizer,
            dropout=dropout,
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.add = tf.keras.layers.Add()
        self.act2 = tf.keras.layers.Activation("relu")

        if type == 2:
            self.alt_tb = TemporalBlock(
                filters=filters,
                kernel_size=1,
                strides=2,
                kernel_regularizer=kernel_regularizer,
                dropout=dropout,
            )
            self.alt_bn = tf.keras.layers.BatchNormalization()
            self.alt_act = tf.keras.layers.Activation("relu")

    
    def call(self, inputs):
        x = self.tb1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.tb2(x)
        x = self.bn2(x)
        if self.type == 2:
            y = self.alt_tb(inputs)
            y = self.alt_bn(y)
            y = self.alt_act(y)
        else:
            y = inputs
        x = self.add([y, x])
        return self.act2(x)
    

class TCResNet(tf.keras.Model):
    def __init__(
        self,
        num_classes: int,
        num_blocks: int,
        add_block_type_1_in_between: bool,
        kernel_size: int,
        channels: list,
        kernel_regularizer = None,
        dropout: float = 0.0,
    ):
        super(TCResNet, self).__init__()
        assert len(channels) - 1 == num_blocks
        self.init_tb = TemporalBlock(
            filters=channels[0],
            kernel_size=3,
            kernel_regularizer=kernel_regularizer,
            dropout=dropout,
        )
        self.temp_blocks = []
        for i in range(num_blocks):
            self.temp_blocks.append(
                TempResBlock(
                    type=2,
                    filters=channels[i],
                    kernel_size=kernel_size,
                    kernel_regularizer=kernel_regularizer,
                    dropout=dropout,
                )
            )
            if add_block_type_1_in_between:
                self.temp_blocks.append(
                    TempResBlock(
                        type=1,
                        filters=channels[i],
                        kernel_size=kernel_size,
                        kernel_regularizer=kernel_regularizer,
                        dropout=dropout,
                    )
                )
        self.avg_pool = tf.keras.layers.AveragePooling2D(pool_size=(2, 1))
        self.flatten = tf.keras.layers.Flatten()
        self.classifier = tf.keras.layers.Dense(
            units=num_classes,
            kernel_regularizer=kernel_regularizer,
            activation="softmax",
        )


    def call(self, inputs):
        # Reshape input to [batch, time, 1, freq]
        x = tf.expand_dims(inputs, axis=2)
        x = self.init_tb(x)
        for tb in self.temp_blocks:
            x = tb(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        return self.classifier(x)