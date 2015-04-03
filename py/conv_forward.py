import datetime
import os
import time

import h5py

from data_handler import *
import cudamat as cm
from cudamat import cudamat_conv_gemm as cc_gemm
from logging_util import initialize_logger


logger = initialize_logger(__name__, '.')


def Save(filename, data_dict):
    logger.info('Saving model to {}'.format(filename))
    f = h5py.File(filename, 'w')
    for key, value in data_dict.items():
        dset = f.create_dataset(key, value.shape, value.dtype)
        dset[:, :] = value
    f.close()


def Update(w, dw, dw_history, momentum, eps, l2_decay):
    dw_history.mult(momentum)
    if l2_decay != 0:
        dw.add_mult(w, l2_decay)
    dw_history.add_mult(dw, -eps)
    w.add(dw_history)


def Encode(data_handle, model_filename, image_prefix, num_filters):
    num_filters = num_filters
    kernel_size_y = 7
    kernel_size_x = 7
    stride_y = 1
    stride_x = 1
    padding_y = 3
    padding_x = 3

    batch_size, image_size_x, image_size_y, num_input_channels = data_handle.GetBatchShape()

    conv_desc = cm.GetConvDesc(num_input_channels, num_filters,
                               kernel_size_y, kernel_size_x, stride_y,
                               stride_x, padding_y, padding_x)
    images_shape = (batch_size, image_size_x, image_size_y, num_input_channels)
    output_shape = cm.GetOutputShape4D(images_shape, conv_desc)
    filters_shape = (
        conv_desc.num_output_channels, conv_desc.kernel_size_x, conv_desc.kernel_size_y, conv_desc.num_input_channels)

    v = cm.empty(images_shape)

    # btch_size, neur_x neur_y, num_filters
    h = cm.empty(output_shape)
    deriv_h = cm.empty(output_shape)

    with h5py.File(model_filename, 'r') as f:
        w_enc = cm.CUDAMatrix(f['w_enc'][()])
        b_enc = cm.CUDAMatrix(f['b_enc'][()])

    w_enc.set_shape4d(filters_shape)

    h.assign(0)
    deriv_h.assign(0)
    v.assign(0)

    data_handle.GetBatch(v)
    cc_gemm.convUp(v, w_enc, h, conv_desc)
    cc_gemm.AddAtAllLocs(h, b_enc)
    h.lower_bound(0)
    # h.apply_sigmoid()


def main():
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')

    data_path = os.path.expanduser(os.path.join('~', 'goback', 'data'))
    dataset_path = os.path.join(data_path, 'coco', 'images')

    data_filename = os.path.join(dataset_path, 'train64.h5')
    mean_filename = os.path.join(dataset_path, 'train64_statistics.h5')

    batch_size = 10000
    data_handle = DataHandler(data_filename, mean_filename, 64, 64, 64, 64, batch_size, batch_size)

    model_filename = 'ae_5_32_%s_model.h5' % st
    image_prefix = 'ae_5_32_%s' % st
    Encode(data_handle, model_filename, image_prefix, 32)

    logger.info('Final losses - 32: {}, 64: {}'.format(loss32, loss64))


if __name__ == '__main__':
    # pdb.set_trace()
    # board = LockGPU()
    # print 'Using board', board
    # ...
    # FreeGPU(board)
    board = -1
    try:
        # cm.cuda_set_device(2)
        # cm.cublas_init()
        board = LockGPU()
        cm.CUDAMatrix.init_random(0)
        main()
    finally:
        if board > -1:
            # cm.cublas_shutdown()
            FreeGPU(board)
