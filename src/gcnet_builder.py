from keras import backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Conv3D, Conv3DTranspose
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import add
from keras.layers.core import Lambda
from cost_volume import cost_volume


def _bn_relu_conv2d(channels):
    def _f(input_):
        x = BatchNormalization(axis=1)(input_)
        x = Activation('relu')(x)
        # x = Activation('relu')(input_)
        x = Conv2D(channels, (3, 3), strides=(1, 1), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(x)
        return x
    return _f


def _conv3d_bn_relu(channels=32, strides=(1, 1, 1)):
    def _f(input_):
        x = Conv3D(filters=channels, kernel_size=(3, 3, 3), strides=strides, padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(input_)
        x = BatchNormalization(axis=1)(x)
        x = Activation('relu')(x)
        return x
    return _f


def _residual_unit(n_in_features, n_out_features, stride):
    def _f(input_):
        if n_in_features == n_out_features:
            identity = input_

            x = _bn_relu_conv2d(n_in_features)(input_)
            x = _bn_relu_conv2d(n_in_features)(x)

            out = add([x, identity])
        else:
            x = BatchNormalization(axis=1)(input_)
            x = Activation('relu')(x)
            # x = Activation('relu')(input_)

            identity = x

            x = Conv2D(n_out_features, (3, 3), strides=stride, padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(x)
            x = _bn_relu_conv2d(n_out_features)(x)

            identity = Conv2D(n_out_features, (1, 1), strides=stride, padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(identity)

            out = add([x, identity])

        return out
    return _f


def _residual_block(n_in_features, n_out_features, n_layers, stride):
    def _f(input_):
        for i in range(n_layers):
            if i == 0:
                x = _residual_unit(n_in_features, n_out_features, stride)(input_)
            else:
                x = _residual_unit(n_out_features, n_out_features, (1, 1))(x)
        return x
    return _f


def _unary():
    def _f(input_):
        # Conv1
        x = Conv2D(32, (5, 5), strides=(2, 2), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(input_)

        # Conv2 to Conv18: 8 residual blocks, each block being 2x residual unit with shortcut
        resnet_depth = 8
        x = _residual_block(32, 32, resnet_depth, (3, 3))(x)
        x = _bn_relu_conv2d(32)(x)
        return x
    return _f


def _get_unary_model(input_shape):
    unary_input = Input(shape=input_shape)
    unary_output = _unary()(unary_input)
    model = Model(unary_input, unary_output)
    return model


def _conv3d_unit(channels=32):
    def _f(input_):
        x = _conv3d_bn_relu(channels=channels, strides=(1, 1, 1))(input_)
        x = _conv3d_bn_relu(channels=channels, strides=(1, 1, 1))(x)
        return x
    return _f


def _soft_arg_min(input_, dmax):
    """softargmin = Sum(d x softmax(-cost[d])) for d=0,...,D
    input_ has shape [DxHxWx1]"""
    x = K.squeeze(input_, axis=1)
    x = K.permute_dimensions(x, (0, 2, 3, 1))
    softmax = K.softmax(x)
    softmax = K.permute_dimensions(softmax, (0, 1, 3, 2))
    disparities = K.expand_dims(K.arange(dmax, dtype='float32'), axis=0)
    output = K.dot(disparities, softmax)
    return K.squeeze(output, axis=0)


class GCNetBuilder(object):
    def __init__(self, dmax):
        self.model = None
        self.max_disparity = dmax

    def build(self, input_image_shape=None):
        left_input = Input(shape=input_image_shape)
        right_input = Input(shape=input_image_shape)

        # Unaries
        unary_model = _get_unary_model(input_image_shape)
        left_unary = unary_model(left_input)
        right_unary = unary_model(right_input)
        
        # Cost Volume Layer
        cv = Lambda(cost_volume, arguments={'dmax': self.max_disparity//2})([left_unary, right_unary])

        # Context learning: 3D convolutions
        n_32 = 32
        n_64 = 64
        n_128 = 128
        layer_19 = _conv3d_bn_relu(channels=n_32, strides=(1, 1, 1))(cv)
        layer_20 = _conv3d_bn_relu(channels=n_32, strides=(1, 1, 1))(layer_19)
        
        layer_21 = _conv3d_bn_relu(channels=n_64, strides=(2, 2, 2))(cv)
        layer_22 = _conv3d_bn_relu(channels=n_64, strides=(1, 1, 1))(layer_21)
        layer_23 = _conv3d_bn_relu(channels=n_64, strides=(1, 1, 1))(layer_22)
        
        layer_24 = _conv3d_bn_relu(channels=n_64, strides=(2, 2, 2))(layer_21)
        layer_25 = _conv3d_bn_relu(channels=n_64, strides=(1, 1, 1))(layer_24)
        layer_26 = _conv3d_bn_relu(channels=n_64, strides=(1, 1, 1))(layer_25)
        
        layer_27 = _conv3d_bn_relu(channels=n_64, strides=(2, 2, 2))(layer_24)
        layer_28 = _conv3d_bn_relu(channels=n_64, strides=(1, 1, 1))(layer_27)
        layer_29 = _conv3d_bn_relu(channels=n_64, strides=(1, 1, 1))(layer_28)

        layer_30 = _conv3d_bn_relu(channels=n_128, strides=(2, 2, 2))(layer_27)
        layer_31 = _conv3d_bn_relu(channels=n_128, strides=(1, 1, 1))(layer_30)
        layer_32 = _conv3d_bn_relu(channels=n_128, strides=(1, 1, 1))(layer_31)

        # Context learning: 3D Deconvolutions
        layer_33 = Conv3DTranspose(n_64, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(layer_32)
        x = add([layer_33, layer_29])

        layer_34 = Conv3DTranspose(n_64, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(x)
        x = add([layer_34, layer_26])

        layer_35 = Conv3DTranspose(n_64, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(x)
        x = add([layer_35, layer_23])

        layer_36 = Conv3DTranspose(n_32, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(x)
        x = add([layer_36, layer_20])

        layer_37 = Conv3DTranspose(1, (3, 3, 3), strides=(2, 2, 2), padding='same', data_format='channels_first', kernel_initializer='glorot_normal')(x)

        # Soft argmin
        layer_38 = Lambda(_soft_arg_min, arguments={'dmax': self.max_disparity})(layer_37)

        self.model = Model(inputs=[left_input, right_input], outputs=layer_38)
        
        return self.model
