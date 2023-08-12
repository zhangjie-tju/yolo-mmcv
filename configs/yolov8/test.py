custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
model = dict(type='YOLOV5',
             pretrained=None,
             backbone=dict(type='CSPDarknetV8', deepen_factor=0.33, widen_factor=0.5),
             neck=dict(type='YOLOV8Neck', deepen_factor=0.33, widen_factor=0.5),
             bbox_head=dict(type='YOLOV8Detect',
                            num_classes=1,
                            in_channels=[256, 512, 1024],
                            widen_factor=0.5,
                            reg_max=16,
                            norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                            act_cfg=dict(type='SiLU', inplace=True),
                            featmap_strides=[8, 16, 32],
                            prior_generator=dict(
            type='mmdet.MlvlPointGenerator', offset=0.5, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            reduction='none',
            loss_weight=0.5),
        loss_bbox=dict(
            type='IoULoss',
            iou_mode='ciou',
            bbox_format='xyxy',
            reduction='sum',
            loss_weight=7.5,
            return_iou=False),
        loss_dfl=dict(
            type='mmdet.DistributionFocalLoss',
            reduction='mean',
            loss_weight=0.375)
                            ),
        train_cfg=dict(
            assigner=dict(
                type='BatchTaskAlignedAssigner',
                num_classes=1,
                use_ciou=True,
                topk=10,
                alpha=0.5,
                beta=6.0,
                eps=1e-09)),
        test_cfg=dict(
            multi_label=True,
            nms_pre=30000,
            score_thr=0.001,
            nms=dict(type='nms', iou_threshold=0.7),
            max_per_img=300)
            )
data_root = '../datasets/cat/'
dataset_type = 'CocoDataset'
img_norm_cfg = dict(mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True)
img_scale = (640, 640)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='MultiScaleFlipAug',
         img_scale=(640, 640),
         flip=False,
         transforms=[
             dict(type='Resize', keep_ratio=True),
             dict(type='Normalize', mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True),
             dict(type='Pad', size_divisor=32),
             dict(type='ImageToTensor', keys=['img']),
             dict(type='Collect',
                  keys=['img'],
                  meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
                             'flip_direction', 'img_norm_cfg'))
         ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'images/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            # dict(type='Mosaic', img_scale=img_scale, pad_val=114.0),
            # dict(type='Expand', mean=[0, 0, 0], to_rgb=True, ratio_range=(1, 2)),
            # dict(type='MinIoURandomCrop', min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9), min_crop_size=0.3),
            dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
            # dict(type='RandomFlip', flip_ratio=0.5),
            # dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect',
                 keys=['img', 'gt_bboxes', 'gt_labels'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor',
                            'img_norm_cfg'))
        ]),
    val=dict(type=dataset_type,
             ann_file=data_root + 'annotations/test.json',
             img_prefix=data_root + 'images/',
             pipeline=[
                 dict(type='LoadImageFromFile'),
                 dict(type='MultiScaleFlipAug',
                      img_scale=(640, 640),
                      flip=False,
                      transforms=[
                          dict(type='Resize', keep_ratio=True),
                          dict(type='Normalize', mean=[0, 0, 0], std=[255.0, 255.0, 255.0], to_rgb=True),
                          dict(type='Pad', size_divisor=32),
                          dict(type='ImageToTensor', keys=['img']),
                          dict(type='Collect',
                               keys=['img'],
                               meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape',
                                          'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'))
                      ])
             ]))
optimizer = dict(type='SGD',
                 lr=0.001,
                 momentum=0.937,
                 weight_decay=0.0005,
                 nesterov=True,
                 paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(policy='step', warmup='linear', warmup_iters=2000, warmup_ratio=0.1, step=[150, 250])
runner = dict(type='EpochBasedRunner', max_epochs=400)
evaluation = dict(interval=20, metric=['bbox'])
checkpoint_config = dict(interval=100)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])
