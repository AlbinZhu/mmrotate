'''
Author: bin.zhu
Date: 2022-06-24 15:25:26
LastEditors: bin.zhu
LastEditTime: 2022-06-28 09:43:31
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
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=20.0),
        num_classes=2))

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

runner = dict(type='EpochBasedRunner', max_epochs=50)