dataset_type = 'DOTADataset'
data_root="/data/user-njf86/DOTA/DOTA1.0-1.5/DOTA1.0/dota_devkit/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_label=False),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    # dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes']),
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        task='Task1',
        ann_file=data_root + 'trainval/labelTxt/',
        img_prefix=data_root + 'trainval/images/',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        task='Task1',
        ann_file=data_root + 'test/labelTxt/',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline))
evaluation = None
