class SpatialAttention(Layer):
    """
     Spatial Attention Mechanism.
    """

    def __init__(self, bias=False, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.bias = bias
        # Define the convolution layer to apply channel-wise attention
        self.conv = Conv2D(
            filters=32,
            kernel_size=(7, 7),
            strides=(1, 1),
            padding='same',
            use_bias=self.bias,
            kernel_initializer='he_normal'
        )

    def attention_layer(self, x):
        # Compute max and average pooling along the channel axis
        max_pool = Lambda(lambda z: tf.reduce_max(z, axis=1, keepdims=True))(x)
        avg_pool = Lambda(lambda z: tf.reduce_mean(z, axis=1, keepdims=True))(x)

        # Concatenate max and average pooling results along the channel axis
        concat = Concatenate(axis=1)([max_pool, avg_pool])

        # Apply convolution to the concatenated result
        output = self.conv(concat)

        # Apply sigmoid activation
        output = tf.sigmoid(output)

        # Apply channel-wise attention
        output = output * x

        return output
