from keras import backend as K
from keras.losses import mean_absolute_error
import numpy as np


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))

def sparse_mean_absolute_error(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, 0.), K.floatx())
    sparse_mae = K.abs(mask_true * (y_true - y_pred))
    # masked_mae = K.sum(sparse_mae, axis=-1) / K.sum(mask_true, axis=-1)
    masked_mae = K.sum(sparse_mae) / K.sum(mask_true)
    return masked_mae


# y_ref = np.array( [[2., 1., 3.],
#         [0.5, -1., 2.5],
#         [-1., -0.5, 2.]])
#
# y_ref_sparse = np.array( [[0., 1., 3.],
#         [0.5, 0., 2.5],
#         [-1., -0.5, 2.]])
#
# # y_pred = np.array( [[2., 1., 3.],
# #         [0.5, -1., 2.5],
# #         [-1., -0.5, 2.]])
#
# # y_pred = np.array( [[[1.5, 1.5, 2.5],
# #         [1., -1.5, 2.0],
# #         [-1.5, -1., 2.5]]])
#
# y_pred = np.array( [[[1.0, 1.5, 2.5],
#         [1., -1.5, 2.5],
#         [-1.5, -1.5, 2.5]]])
#
# y_ref = np.expand_dims(y_ref, axis=0)
# y_ref_sparse = np.expand_dims(y_ref_sparse, axis=0)
# y_pred = np.expand_dims(y_pred, axis=0)
#
#
# ky_ref = K.variable(y_ref)
# ky_ref_sparse = K.variable(y_ref_sparse)
# ky_pred = K.variable(y_pred)
#
# kdelta = sparse_mean_absolute_error(ky_ref, ky_pred)
# kdelta_sparse = sparse_mean_absolute_error(ky_ref_sparse, ky_pred)
#
# delta = K.eval(kdelta)
# print(delta)
#
# delta_sparse = K.eval(kdelta_sparse)
# print(delta_sparse)
