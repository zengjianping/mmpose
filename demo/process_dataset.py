import os, sys, time, json
import numpy as np
from argparse import ArgumentParser
from typing import Dict

from mmpose.apis.inferencers import MMPoseInferencer, get_model_aliases


filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)


def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--show-alias', action='store_true',
        help='Display all the available model aliases.')
    parser.add_argument('--work-mode', type=int, default=0, help='Work mode')
    parser.add_argument('--input-path', type=str, default=None, help='input path')
    parser.add_argument('--output-path', type=str, default=None, help='output path')

    # init args
    parser.add_argument('--pose2d', type=str, default=None,
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument('--pose2d-weights', type=str, default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose2d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument('--pose3d', type=str, default=None,
        help='Pretrained 3D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument('--pose3d-weights', type=str, default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose3d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument('--det-model', type=str, default=None,
        help='Config path or alias of detection model.')
    parser.add_argument('--det-weights', type=str, default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument('--det-cat-ids', type=int, nargs='+', default=0,
        help='Category id for detection model.')
    parser.add_argument('--scope', type=str, default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument('--device', type=str, default=None,
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')
    parser.add_argument('--show-progress', action='store_true',
        help='Display the progress bar during inference.')

    # call args
    parser.add_argument('--inputs', type=str, default=None,
        help='Input image/video path or folder path.')
    parser.add_argument('--show', action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument('--draw-bbox', action='store_true',
        help='Whether to draw the bounding boxes.')
    parser.add_argument('--draw-heatmap', action='store_true', default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument('--bbox-thr', type=float, default=filter_args['bbox_thr'],
        help='Bounding box score threshold')
    parser.add_argument('--nms-thr', type=float, default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')
    parser.add_argument('--pose-based-nms', type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'], help='Whether to use pose-based NMS')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
        help='Keypoint score threshold')
    parser.add_argument('--tracking-thr', type=float, default=0.3,
        help='Tracking threshold')
    parser.add_argument('--use-oks-tracking', action='store_true',
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument('--disable-norm-pose-2d', action='store_true',
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument('--disable-rebase-keypoint', action='store_true', default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument('--num-instances', type=int, default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument('--radius', type=int, default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument('--thickness', type=int, default=1,
        help='Link thickness for visualization.')
    parser.add_argument('--skeleton-style', default='mmpose', type=str,
        choices=['mmpose', 'openpose'], help='Skeleton style selection')
    parser.add_argument('--black-background', action='store_true',
        help='Plot predictions on a black image')
    parser.add_argument('--vis-out-dir', type=str, default='',
        help='Directory for saving visualized results.')
    parser.add_argument('--pred-out-dir', type=str, default='',
        help='Directory for saving inference results.')

    # The default arguments for prediction filtering differ for top-down
    # and bottom-up models. We assign the default arguments according to the
    # selected pose2d model
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if args.pose2d is not None and model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break

    arg_dict = vars(parser.parse_args())
    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights',
        'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = arg_dict.pop(init_kw)
    
    call_kws = [
        'inputs', 'show', 'draw_bbox', 'draw_heatmap', 'bbox_thr',
        'nms_thr', 'pose_based_nms', 'kpt_thr', 'tracking_thr',
        'use_oks_tracking', 'disable_norm_pose_2d', 'disable_rebase_keypoint',
        'num_instances', 'radius', 'thickness', 'skeleton_style', 'black_background',
        'vis_out_dir', 'pred_out_dir'
    ]
    call_args = {}
    for call_kw in call_kws:
        call_args[call_kw] = arg_dict.pop(call_kw)

    return args, init_args, call_args


def display_model_aliases(model_aliases: Dict[str, str]) -> None:
    """Display the available model aliases and their corresponding model
    names."""
    aliases = list(model_aliases.keys())
    max_alias_length = max(map(len, aliases))
    print(f'{"ALIAS".ljust(max_alias_length+2)}MODEL_NAME')
    for alias in sorted(aliases):
        print(f'{alias.ljust(max_alias_length+2)}{model_aliases[alias]}')

def test_pose_detector(init_args, call_args):
    inferencer = MMPoseInferencer(**init_args)
    result_generator = inferencer(**call_args)
    for result in result_generator:
        print(result)

def process_coco_dataset(input_file, output_file, init_args, call_args):
    data_dir = call_args['inputs']

    coco_data = json.loads(open(input_file, 'r').read())
    coco_data.pop('annotations')

    inferencer = MMPoseInferencer(**init_args)

    annot_id = 0
    category_id = 0
    for category in coco_data['categories']:
        if category['name'] == 'golf_pose':
            category_id = category['id']

    annotations = list()
    image_infos = coco_data['images']

    for image_idx, image_info in enumerate(image_infos):
        print(f'Processing {image_idx+1}/{len(image_infos)}...')
        image_file = os.path.join(data_dir, image_info['file_name'])
        call_args['inputs'] = image_file
        result_generator = inferencer(**call_args)

        for result in result_generator:
            prediction = result['predictions'][0]
            #print(prediction)

            for region in prediction:
                if region['bbox_score'] < 0.5:
                    continue
                annot_id += 1
                bbox = region['bbox'][0].copy()
                bbox[2] = (bbox[2] - bbox[0])
                bbox[3] = (bbox[3] - bbox[1])
                kp_coords = region['keypoints']
                kp_scores = region['keypoint_scores']

                keypoints = list()
                for kp_coord, kp_score in zip(kp_coords, kp_scores):
                    keypoint = kp_coord + [2] if kp_score >= 0.5 else [0, 0, 0]
                    keypoints.extend(keypoint)
                keypoints.extend([0, 0, 0] * 4)

                annot_obj = {
                    "id": annot_id,
                    "image_id": image_info['id'],
                    "category_id": category_id,
                    "segmentation": [],
                    "area": bbox[2] * bbox[3],
                    "bbox": bbox,
                    "iscrowd": 0,
                    "attributes": {
                        "occluded": False,
                        "keyframe": False
                    },
                    "keypoints": keypoints,
                    "num_keypoints": len(keypoints) // 3
                }
                annotations.append(annot_obj)
    coco_data['annotations'] = annotations

    json_str = json.dumps(coco_data, ensure_ascii=False, indent=4)
    open(output_file, 'w').write(json_str)

def main():
    args, init_args, call_args = parse_args()
    if args.show_alias:
        model_alises = get_model_aliases(init_args['scope'])
        display_model_aliases(model_alises)
    elif args.work_mode == 0:
        test_pose_detector(init_args, call_args)
    elif args.work_mode == 1:
        process_coco_dataset(args.input_path, args.output_path, init_args, call_args)

if __name__ == '__main__':
    main()
