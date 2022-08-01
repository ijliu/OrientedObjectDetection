dataset_type = 'DOTADataset'
data_root="/data/user-njf86/DOTA/DOTA1.0-1.5/DOTA1.0/dota_devkit/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadOrientedAnnotations', with_bbox=True, with_label=False),
    dict(type='RResize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        # task='Task1',
        ann_file=data_root + 'trainval/labelTxt/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # task='Task1',
        ann_file=data_root + 'val/labelTxt/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # task='Task1',
        ann_file=data_root + 'val/labelTxt/',
        img_prefix=data_root + 'val/images/',
        pipeline=test_pipeline))
# evaluation = None
evaluation = dict(interval=12)
