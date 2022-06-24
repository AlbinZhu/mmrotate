'''
Author: bin.zhu
Date: 2022-06-24 15:25:26
LastEditors: bin.zhu
LastEditTime: 2022-06-24 17:06:03
Description: file content
'''
_base_ = [
    '../rotated_retinanet/rotated_retinanet_obb_r50_fpn_1x_dota_le135.py'
]
dataset_type = 'DOTABookDataset'
classes = ('left', 'right')
model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))

data_root = '/albin/page_label/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'train/',
        img_prefix=data_root + 'train'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'val/',
        img_prefix=data_root + 'val/'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'val/',
        img_prefix=data_root + 'val/'))

# 2. model settings
model = dict(
    bbox_head=dict(
        type='RotatedRetinaHead',
        # explicitly over-write all the `num_classes` field from default 15 to 5.
        num_classes=2))