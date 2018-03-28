from keras import backend as K
# from keras.layers.core import Lambda
# from keras.layers.core import Reshape


def _concat_features(lf, states):
    b,f,h,w = lf.get_shape().as_list()
    rf = states[0]
    rfs = rf[:, :, :, :-1]
    disp_rfs = K.spatial_2d_padding(rfs, padding=((0, 0), (1, 0)), data_format='channels_first')
    concat = K.concatenate([lf, rf], axis=2)
    output = K.reshape(concat, (-1, 2*f, h, w))
    return output, [disp_rfs]

def cost_volume(inputs, dmax):
    left_feature = inputs[0]
    right_feature = inputs[1]
    left_feature = K.expand_dims(left_feature, axis=1)
    left_feature = K.repeat_elements(left_feature, dmax, axis=1)
    l,o,n = K.rnn(_concat_features, inputs=left_feature, initial_states=[right_feature], unroll=True)
    return K.permute_dimensions(o, (0, 2, 1, 3, 4))

# class CostVolumeBuilder():
#     @classmethod
#     def get_layer(cls, D):
#         return Lambda(cost_volume, arguments = {'d':D/2})
