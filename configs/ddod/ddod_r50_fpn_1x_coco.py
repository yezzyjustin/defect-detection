_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
input_size = 512
model = dict(
    type='DDOD',
    backbone=dict(
            type='RegNet',
            arch='regnetx_800mf',
            out_indices=(0, 1, 2, 3),
            frozen_stages=1,
            norm_cfg=dict(type='BN', requires_grad=True),
            norm_eval=True,
            style='pytorch',
            init_cfg=dict(
                type='Pretrained', checkpoint='open-mmlab://regnetx_800mf')),
    neck=[

            dict(
                # type='HighFPNRetinanet',
                type='FPN',
                in_channels=[64, 128, 288, 672],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_input',
                num_outs=5,
            ),

            dict(
                type='BFP',
                in_channels=256,
                num_levels=5,
                refine_level=2,
                refine_type='non_local')
        ],

    bbox_head=dict(
        type='DDODHead',
        num_classes=6,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,

        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),




        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),

        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_iou=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),


    train_cfg=dict(
        # assigner is mean cls_assigner
        assigner=dict(type='ATSSAssigner', topk=9, alpha=0.8),
        reg_assigner=dict(type='ATSSAssigner', topk=9, alpha=0.5),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

# This `persistent_workers` is only valid when PyTorch>=1.7.0
data = dict(persistent_workers=True)
# optimizer
optimizer = dict(type='SGD', lr=0.00025, momentum=0.9, weight_decay=0.0001)














