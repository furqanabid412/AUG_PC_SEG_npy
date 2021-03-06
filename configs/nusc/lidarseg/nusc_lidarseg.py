import itertools
import logging

from det3d.utils.config_tool import get_downsample_factor


# in case of complete dataset v1.0-trainval

# rate = 0.40
# nsweeps = 0
# data_root = 'F:/Datasets/nuscenes' # myPC
# # add the dataroot information , if running on server
#
# db_info_path = "/dbinfos_{:03d}rate_{:02d}sweeps_withvelo_crossmodal.pkl".format(int(rate*100), nsweeps)
# train_anno = "/infos_train_{:02d}sweeps_withvelo_filter_True_{:03d}rate_crossmodal.pkl".format(nsweeps, int(rate*100))
# val_anno = "/infos_val_00sweeps_withvelo_filter_True_crossmodal.pkl"
# test_anno = ""
# version = 'v1.0-trainval'

# in case of mini dataset v1.0-mini
rate = 0.15
nsweeps = 0
# data_root = 'E:/Datasets/NuScenes/new_test_run/v1.0-mini'
# data_root = '/media/furqan/Terabyte/Lab/datasets/Nuscenes/v1.0-mini'
data_root = '/usb/ssd512/nusc/data'
db_info_path = "/dbinfos_{:03d}rate_{:02d}sweeps_withvelo_crossmodal.pkl".format(int(rate*100), nsweeps)

train_anno = "/infos_train_{:02d}sweeps_withvelo_filter_True_{:03d}rate_crossmodal.pkl".format(nsweeps, int(rate*100))
val_anno = "/infos_val_00sweeps_withvelo_filter_True_crossmodal.pkl"
test_anno = ""
version = 'v1.0-trainval'



use_img = True
use_aug = True
DOUBLE_FLIP = True

voxel_size = [0.8, 0.8, 8]
pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

fade_epoch = 15

num_input_features = 5 + int(use_img) * 64

tasks = [
    dict(num_class=1, class_names=["car"]),
    dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    dict(num_class=2, class_names=["bus", "trailer"]),
    dict(num_class=1, class_names=["barrier"]),
    dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="PPFusion",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64, 64],
        num_input_features=num_input_features,
        with_distance=False,
        voxel_size=voxel_size,
        pc_range=pc_range,
    ),
    img_backbone=dict(
        type="DLASeg",  # ResNet18, DLASeg
    ) if use_img else None,
    backbone=dict(type="PointPillarsScatter", ds_factor=1, num_input_features=64),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="CenterHead",
        in_channels=sum([128, 128, 128]),
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)}, # (output_channel, num_conv)
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    nms=dict(
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=pc_range[:2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=voxel_size[:2],
    double_flip=DOUBLE_FLIP
)


# dataset settings
dataset_type = "NuScenesDataset"

db_num = 1
db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path=data_root+db_info_path,
    sample_groups=[
        dict(car=1*db_num),
        dict(truck=1*db_num),
        dict(construction_vehicle=1*db_num),
        dict(bus=1*db_num),
        dict(trailer=1*db_num),
        dict(barrier=1*db_num),
        dict(motorcycle=1*db_num),
        dict(bicycle=1*db_num),
        dict(pedestrian=1*db_num),
        dict(traffic_cone=1*db_num),
    ],
    # implemented in builder.py , check the sampled dictionary contains lidar
    # points more than those mentioned below for each class
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                truck=5,
                bus=5,
                trailer=5,
                construction_vehicle=5,
                traffic_cone=5,
                barrier=5,
                motorcycle=5,
                bicycle=5,
                pedestrian=5,
            )
        ),
        # in current implementation, filter sampled(annotation for augmentations)objects
        # by difficulty is not implemented

        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
) if use_aug else None

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    db_sampler=db_sampler,
    class_names=class_names,
    use_img=use_img,
    remove_points_after_sample=True,  # False
    doLidarSegmentation = True,
    postAugmentation = True,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    use_img=use_img,
)

voxel_generator = dict(
    range=pc_range,
    voxel_size=voxel_size,
    max_points_in_voxel=20,
    max_voxel_num=[30000, 60000],
    double_flip=DOUBLE_FLIP,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, use_img=use_img),
    dict(type="LoadLidarSegLabels", dataset= 'NUSCENES', doAugmentation=True),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, use_img=use_img),
    # dict(type="Preprocess", cfg=train_preprocessor),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type, use_img=use_img),
    dict(type="LoadLidarSegLabels", dataset='NUSCENES', doAugmentation=True),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, use_img=use_img),
    # dict(type="Preprocess", cfg=train_preprocessor),
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root+train_anno,
        ann_file=data_root+train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
        double_flip=DOUBLE_FLIP,
        version=version,
        use_img=use_img,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root+val_anno,
        test_mode=True,
        ann_file=data_root+val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        double_flip=DOUBLE_FLIP,
        version=version,
        use_img=use_img,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=data_root+test_anno,
        ann_file=data_root+test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
        double_flip=DOUBLE_FLIP,
        version=version,
        use_img=use_img,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
# yapf:enable
# runtime settings
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [('train', 1)]
