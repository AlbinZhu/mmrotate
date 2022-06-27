'''
Author: bin.zhu
Date: 2022-06-27 11:25:46
LastEditors: bin.zhu
LastEditTime: 2022-06-27 11:39:55
Description: file content
'''
_base_ = ['../rotated_retinanet/rotated_retinanet_hbb_r50_fpn_1x_dota_oc.py']

dataset_type = 'DOTABookDataset'
classes = ('left', 'right')
model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(
            type='GDLoss', loss_type='gwd', loss_weight=5.0, num_classes=2)))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data_root = '/albin/page_label/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file=data_root + 'train/',
        img_prefix=data_root + 'train',
        pipeline=train_pipeline),
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
