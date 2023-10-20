# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


_base_ = [
    # '../../_base_/models/cascade_mask_rcnn_convnext_fpn.py',
    '../../_base_/models/cascade_rcnn_convnext_fpn.py',
    # '../../_base_/datasets/coco_instance.py',
    '../../_base_/datasets/coco_detection.py',
    '../../_base_/schedules/schedule_1x.py', '../../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        in_chans=3,
        depths=[3, 3, 9, 3], 
        dims=[96, 192, 384, 768], 
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3],
        init_cfg=None,
    ),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=[
                dict(type="Shared2FCBBoxHead", num_classes=1),
                dict(type="Shared2FCBBoxHead", num_classes=1),
                dict(type="Shared2FCBBoxHead", num_classes=1),
            ]
    ),
)

# classes
classes = ("car",)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

image_size = (1024, 1024)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=image_size),  # padding to image_size leads 0.5+ mAP
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1024),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

# data = dict(train=dict(pipeline=train_pipeline), persistent_workers=True)

# Path to cityscapes and sim10k
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    persistent_workers=True, # Testing this feature
    train=dict(
        type="CocoDataset",
        img_prefix="data/Sim10k/JPEGImages/",
        classes=classes,
        ann_file="data/Sim10k/COCOAnnotations/voc2012_annotations.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CityscapesDataset",
        img_prefix="data/Cityscapes/leftImg8bit/val/",
        classes=classes,
        ann_file="data/Cityscapes/instancesonly_filtered_gtFine_val.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CityscapesDataset",
        img_prefix="data/Cityscapes/leftImg8bit/val/",
        classes=classes,
        ann_file="data/Cityscapes/instancesonly_filtered_gtFine_val.json",
        pipeline=test_pipeline,
    ),
)

optimizer = dict(
    _delete_=True, type="AdamW", lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05,
)

lr_config = dict(warmup_iters=1000, step=[27, 33])
runner = dict(max_epochs=36)

# you need to set mode='dynamic' if you are using pytorch<=1.5.0
fp16 = dict(loss_scale=dict(init_scale=512))