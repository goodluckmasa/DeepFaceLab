from core.leras import nn
tf = nn.tf


class PercpLoss(nn.LayerBase):

    def __init__(self, resolution, input_ch=3, **kwargs):
        vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet',
                                            input_shape=(resolution, resolution, input_ch))
        vgg19.trainable = False
        layer2_2 = vgg19.get_layer('block2_conv2')
        feature_map = tf.keras.activations.relu(layer2_2.output)
        self.model = tf.keras.models.Model(vgg19.input, feature_map)

        super().__init__(**kwargs)

    def __call__(self, y_true, y_pred):
        # Transpose images from NCHW to NHWC
        y_true_t = tf.transpose(tf.cast(y_true, tf.float32), [0, 2, 3, 1])
        y_pred_t = tf.transpose(tf.cast(y_pred, tf.float32), [0, 2, 3, 1])

        # Get feature maps for images
        y_true_feat = self.model(y_true_t) / 12.75
        y_pred_feat = self.model(y_pred_t) / 12.75

        return tf.reduce_mean(tf.keras.losses.MSE(y_true_feat, y_pred_feat))


nn.PercpLoss = PercpLoss
