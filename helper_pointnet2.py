import os
import sys
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
print(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/interpolation'))
from tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from tf_ops.grouping.tf_grouping import query_ball_point, group_point, knn_point
from tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np

from helper_net import Ops as Ops


def tf_cov(x):
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    cov_xx = tf.matmul(tf.transpose(x - mean_x, [0, 1, 3, 2]), x - mean_x) / tf.cast(4 - 1, tf.float32)
    return cov_xx


def expan_tmp(p, b, n):  # expand 1*3 to B*N*3
    tmp = tf.expand_dims(tf.constant(p), axis=0)
    tmp = tf.tile(tmp, [n, 1])
    tmp = tf.tile(tf.expand_dims(tmp, 0), [b, 1, 1])
    return tmp


def cos_tmp_s(tem0, tem1):
    norm_temp0 = tf.norm(tem0, ord=2)
    norm_temp1 = tf.norm(tem1, ord=2)
    in_multiply = tf.reduce_sum(tf.multiply(tem0, tem1), axis=-1)
    return tf.acos(in_multiply / norm_temp1 / norm_temp0)


def rotation(angle, b, n, x=0, y=0, z=0):
    cos_angle = tf.expand_dims(tf.expand_dims(tf.cos(angle), -1), -1)  # b*n*1*1
    sin_angle = tf.expand_dims(tf.expand_dims(tf.sin(angle), -1), -1)
    with tf.device('/gpu:0'):
        if x:
            r_x = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[1., 0., 0., 0., 0., 0., 0., 0., 0.]]), 0), [b * n, 1, 1]),
                [b, n, 3, 3])
            cos_x = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[0., 0., 0., 0., 1., 0., 0., 0., 1.]]), 0), [b * n, 1, 1]),
                [b, n, 1, 9])
            sin_x = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[0., 0., 0., 0., 0., 1., 0., -1., 0.]]), 0), [b * n, 1, 1]),
                [b, n, 1, 9])
            cos_x_r = tf.reshape(tf.matmul(cos_angle, cos_x), [b, n, 3, 3])  # b*n*1*9 to b*n*3*3
            sin_x_r = tf.reshape(tf.matmul(sin_angle, sin_x), [b, n, 3, 3])
            ro = r_x + cos_x_r + sin_x_r

        if y:
            r_y = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[0., 0., 0., 0., 1., 0., 0., 0., 0.]]), 0), [b * n, 1, 1]),
                [b, n, 3, 3])
            cos_y = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[1., 0., 0., 0., 0., 0., 0., 0., 1.]]), 0), [b * n, 1, 1]),
                [b, n, 1, 9])
            sin_y = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[0., 0., -1., 0., 0., 0., 1., 0., 0.]]), 0), [b * n, 1, 1]),
                [b, n, 1, 9])
            cos_y_r = tf.reshape(tf.matmul(cos_angle, cos_y), [b, n, 3, 3])  # b*n*1*9 to b*n*3*3
            sin_y_r = tf.reshape(tf.matmul(sin_angle, sin_y), [b, n, 3, 3])
            ro = r_y + cos_y_r + sin_y_r

        if z:
            r_z = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[0., 0., 0., 0., 0., 0., 0., 0., 1.]]), 0), [b * n, 1, 1]),
                [b, n, 3, 3])
            cos_z = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[1., 0., 0., 0., 0., 1., 0., 0., 0.]]), 0), [b * n, 1, 1]),
                [b, n, 1, 9])
            sin_z = tf.reshape(
                tf.tile(tf.expand_dims(tf.constant([[0., 1., 0., -1., 0., 1., 0., 0., 0.]]), 0), [b * n, 1, 1]),
                [b, n, 1, 9])
            cos_z_r = tf.reshape(tf.matmul(cos_angle, cos_z), [b, n, 3, 3])  # b*n*1*9 to b*n*3*3
            sin_z_r = tf.reshape(tf.matmul(sin_angle, sin_z), [b, n, 3, 3])
            ro = r_z + cos_z_r + sin_z_r

    return ro


def select(axis, tem, b, n, axi, angel):
    # axis:b*n*3
    # tem:b*n*3
    # axi:b*n*3
    with tf.device('/gpu:0'):
        norm = tf.expand_dims(tf.linalg.cross(axis, tem), -1)
        axi = tf.expand_dims(axi, 2)
        con = tf.squeeze(tf.matmul(axi, norm)) < 0  # B*N
        return tf.where(con, 2 * math.pi - angel, angel)


def convert_p(vector, b, n):
    with tf.device('/gpu:0'):
        v = tf.squeeze(tf.slice(vector, [0, 0, 2, 0], [b, n, 1, 3]))  # 局部参考系Z轴
        vy = tf.squeeze(tf.slice(vector, [0, 0, 1, 0], [b, n, 1, 3]))  # 局部参考系Y轴
        vy_n = tf.expand_dims(vy, -1)  # [4,1024,3,1]
        vx = tf.squeeze(tf.slice(vector, [0, 0, 0, 0], [b, n, 1, 3]))  # 局部参考系X轴
        vx = tf.expand_dims(vx, -1)

        q = tf.multiply(v, expan_tmp([0., 1., 1.], b, n))  # 中心点在世界坐标系yoz的投影
        z = expan_tmp([0., 0., 1.], b, n)  # 世界坐标系z轴
        alpha = cos_tmp_s(q, z)  # q与z轴夹角
        x = expan_tmp([1., 0., 0.], b, n)  # 世界坐标系x轴
        r = tf.multiply(v, x) + tf.multiply(tf.norm(q, ord=2), z)  # 中心点在世界坐标系zox的投影
        belt = cos_tmp_s(r, z)  # r与z轴夹角
        y = expan_tmp([0., 1., 0.], b, n)  # 世界坐标系y轴
        y = tf.transpose(tf.expand_dims(y, -1), [0, 1, 3, 2])

        convert1 = select(z, q, b, n, x, alpha)  # 根据z轴与q的叉乘与x轴的乘积判断夹角重置夹角alpha大小,得到convert1
        rotation_alpha = rotation(convert1, b, n, 1., 0., 0.)  # 得到绕x旋转convert1的旋转矩阵
        z = tf.transpose(tf.expand_dims(z, -1), [0, 1, 3, 2])
        z_convert1 = tf.squeeze(tf.matmul(z, rotation_alpha))  # 世界坐标系z轴绕x轴旋转convert1后的坐标轴z_convert1
        y_convert1 = tf.matmul(y, rotation_alpha)  # [4,1024,1,3]  世界坐标系y轴绕x轴旋转convert1后的坐标轴y_convert1
        norm_y_convert1 = tf.norm(y_convert1, ord=2)  # y_convert1二范数
        norm_vyn = tf.norm(vy_n, ord=2)  # 局部参考系Y轴二范数
        gama = tf.squeeze(
            tf.acos(tf.matmul(y_convert1, vy_n) / norm_y_convert1 / norm_vyn))  # 第三次旋转角gama，为y_convert1与局部参考系Y轴夹角
        y_convert1 = tf.squeeze(y_convert1)

        convert2 = select(z_convert1, r, b, n, y_convert1,
                          belt)  # 根据z_convert1轴与r的叉乘与y_convert1轴的乘积判断夹角重置夹角belt大小,得到convert2
        rotation_belt = rotation(convert2, b, n, 0., 1., 0.)  # 得到绕y轴旋转convert2的旋转矩阵
        z_convert2 = tf.squeeze(
            tf.matmul(tf.expand_dims(z_convert1, 2), rotation_belt))  # z_convert1绕y_convert1轴旋转convert2后的坐标轴z_convert2
        convert3 = select(y_convert1, vy, b, n, z_convert2,
                          gama)  # 根据y_convert1轴与局部参考系Y轴的叉乘与z_convert2轴的乘积判断夹角重置夹角gama大小,得到convert3
        R = tf.matmul(rotation(convert3, b, n, 0., 0., 1.),
                      tf.matmul(rotation_belt, rotation_alpha))  # 得到世界坐标系到局部参考系的旋转矩阵
    # W=tf.linalg.inv(R)  # 局部参考系到世界参考系的旋转矩阵
    # zero=tf.expand_dims(expan_tmp([0.,0.,0.],b,n),-1)  # 局部坐标系原点
    # print(R)
    return R  # tf.matmul(W,zero)  其中W*zero为世界坐标系原点在局部参考系中的坐标


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))  # (batch_size, npoint, 3)
    #npoint = new_xyz.get_shape()[1].value
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    with tf.device('/gpu:0'):
        grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
        grouped_xyz_cov = tf_cov(tf.transpose(grouped_xyz, [0, 1, 3, 2]))  # 协方差矩阵
        value, _, vector = tf.linalg.svd(grouped_xyz_cov)  # SVD分解
        R = convert_p(vector, 4, npoint)  # 得到世界坐标系到局部参考系的旋转矩阵和世界坐标系原点在局部参考系中的坐标
        # zero=tf.squeeze(zero)  #B*N*3
        zero_new = tf.tile(tf.expand_dims(-new_xyz, 2), [1, 1, nsample, 1])  # 组合得到
        grouped_xyz_new = tf.add(tf.transpose(tf.matmul(R, tf.transpose(grouped_xyz, [0, 1, 3, 2])), [0, 1, 3, 2]),
                                 zero_new)  # 组合local=R*world+zero_new 得到点在局部参考系下的表达
    if points is not None:
        grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
        if use_xyz:
            new_points = tf.concat([grouped_xyz_new, grouped_points],
                                   axis=-1)  # (batch_size, npoint, nample, 3+channel)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz_new

    return new_xyz, new_points, idx, grouped_xyz_new


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    # batch_size = xyz.get_shape()[0].value
    batch_size = tf.shape(input=xyz)[0]
    nsample = xyz.get_shape()[1].value

    # new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    new_xyz = tf.tile(np.array([0, 0, 0], dtype=np.float32).reshape((1, 1, 3)), (batch_size, 1, 1))

    # idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    idx = tf.tile(np.array(range(nsample), dtype=np.float32).reshape((1, 1, nsample)), (batch_size, 1, 1))

    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))  # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2)  # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz


def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope,
                       bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.compat.v1.variable_scope(scope) as sc:
        # Sample and Grouping
        if group_all:
            nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

        # Point Feature Embedding
        if use_nchw: new_points = tf.transpose(a=new_points, perm=[0, 3, 1, 2])
        for i, num_out_channel in enumerate(mlp):
            ####################################
            new_points = Ops.xxlu(
                Ops.conv2d(new_points, k=(1, 1), out_c=num_out_channel, str=1, pad='VALID', name='l' + str(i)),
                label='lrelu')
            # new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],padding='VALID', stride=[1,1],
            #        bn=bn, is_training=is_training, scope='conv%d'%(i), bn_decay=bn_decay, data_format=data_format)

        if use_nchw: new_points = tf.transpose(a=new_points, perm=[0, 2, 3, 1])

        # Pooling in Local Regions
        if pooling == 'max':
            new_points = tf.reduce_max(input_tensor=new_points, axis=[2], keepdims=True, name='maxpool')
        elif pooling == 'avg':
            new_points = tf.reduce_mean(input_tensor=new_points, axis=[2], keepdims=True, name='avgpool')
        elif pooling == 'weighted_avg':
            with tf.compat.v1.variable_scope('weighted_avg'):
                dists = tf.norm(tensor=grouped_xyz, axis=-1, ord=2, keepdims=True)
                exp_dists = tf.exp(-dists * 5)
                weights = exp_dists / tf.reduce_sum(input_tensor=exp_dists, axis=2,
                                                    keepdims=True)  # (batch_size, npoint, nsample, 1)
                new_points *= weights  # (batch_size, npoint, nsample, mlp[-1])
                new_points = tf.reduce_sum(input_tensor=new_points, axis=2, keepdims=True)
        elif pooling == 'max_and_avg':
            max_points = tf.reduce_max(input_tensor=new_points, axis=[2], keepdims=True, name='maxpool')
            avg_points = tf.reduce_mean(input_tensor=new_points, axis=[2], keepdims=True, name='avgpool')
            new_points = tf.concat([avg_points, max_points], axis=-1)

        # [Optional] Further Processing
        if mlp2 is not None:
            if use_nchw: new_points = tf.transpose(a=new_points, perm=[0, 3, 1, 2])
            for i, num_out_channel in enumerate(mlp2):
                ####################################
                new_points = Ops.xxlu(
                    Ops.conv2d(new_points, k=(1, 1), out_c=num_out_channel, str=1, pad='VALID', name='ll' + str(i)),
                    label='lrelu')
                # new_points = tf_util.conv2d(new_points, num_out_channel, [1,1], padding='VALID', stride=[1,1],
                # bn=bn, is_training=is_training, scope='conv_post_%d'%(i), bn_decay=bn_decay,data_format=data_format)

            if use_nchw: new_points = tf.transpose(a=new_points, perm=[0, 2, 3, 1])

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


def pointnet_sa_module_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, is_training, bn_decay, scope,
                           bn=True, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: list of float32 -- search radius in local region
            nsample: list of int32 -- how many points in each local region
            mlp: list of list of int32 -- output size for MLP on each point
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'
    with tf.compat.v1.variable_scope(scope) as sc:
        new_xyz = gather_point(xyz, farthest_point_sample(npoint, xyz))
        new_points_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            grouped_xyz = group_point(xyz, idx)
            grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1, 1, nsample, 1])
            if points is not None:
                grouped_points = group_point(points, idx)
                if use_xyz:
                    grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)
            else:
                grouped_points = grouped_xyz
            if use_nchw: grouped_points = tf.transpose(a=grouped_points, perm=[0, 3, 1, 2])
            for j, num_out_channel in enumerate(mlp_list[i]):
                ####################################
                grouped_points = Ops.xxlu(
                    Ops.conv2d(grouped_points, k=(1, 1), out_c=num_out_channel, str=1, pad='VALID',
                               name='lll' + str(i)), label='lrelu')
                # grouped_points = tf_util.conv2d(grouped_points, num_out_channel, [1,1],
                # padding='VALID', stride=[1,1], bn=bn, is_training=is_training,scope='conv%d_%d'%(i,j), bn_decay=bn_decay)

            if use_nchw: grouped_points = tf.transpose(a=grouped_points, perm=[0, 2, 3, 1])
            new_points = tf.reduce_max(input_tensor=grouped_points, axis=[2])
            new_points_list.append(new_points)
        new_points_concat = tf.concat(new_points_list, axis=-1)
        return new_xyz, new_points_concat


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.compat.v1.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum(input_tensor=(1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            ####################################
            new_points1 = Ops.xxlu(
                Ops.conv2d(new_points1, k=(1, 1), out_c=num_out_channel, str=1, pad='VALID', name='llll' + str(i)),
                label='lrelu')
            # new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],padding='VALID', stride=[1,1],
            #    bn=bn, is_training=is_training,scope='conv_%d'%(i), bn_decay=bn_decay)

        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1
