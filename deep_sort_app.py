# coding=utf-8
# vim: expandtab:ts=4:sw=4
from __future__ import division, print_function, absolute_import

import argparse
import os

import cv2
import numpy as np

from application_util import preprocessing
from application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


def gather_sequence_info(sequence_dir, detection_file):
    """Gather sequence information, such as image filenames, detections,
    groundtruth (if available).
    从MOT检测数据集中获取序列信息

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detection file.

    Returns
    -------
    Dict
        A dictionary of the following sequence information:
        返回下述序列信息dict

        * sequence_name: Name of the sequence 序列名
        * image_filenames: A dictionary that maps frame indices to image
          filenames.图像文件名
        * detections: A numpy array of detections in MOTChallenge format. 使用MOTChallenge格式的np
        * groundtruth: A numpy array of ground truth in MOTChallenge format. MOTChallenge格式的gt
        * image_size: Image size (height, width). 图像大小height，width
        * min_frame_idx: Index of the first frame. 第一帧index
        * max_frame_idx: Index of the last frame. 最后一针index

    """
    image_dir = os.path.join(sequence_dir, "img1") # 图像文件
    # 对应的文件名和绝对路径名
    image_filenames = {
        int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
        for f in os.listdir(image_dir)}
    # gt文件
    groundtruth_file = os.path.join(sequence_dir, "gt/gt.txt")

    # MOT detections结果
    detections = None
    if detection_file is not None:
        detections = np.load(detection_file)
    groundtruth = None
    if os.path.exists(groundtruth_file):
        groundtruth = np.loadtxt(groundtruth_file, delimiter=',')

    if len(image_filenames) > 0:
        # 读取第一帧
        image = cv2.imread(next(iter(image_filenames.values())),
                           cv2.IMREAD_GRAYSCALE)
        image_size = image.shape
    else:
        image_size = None

    if len(image_filenames) > 0:
        min_frame_idx = min(image_filenames.keys()) # 这里image_filenames是0：XX.jpg，1：XX,jpg
        max_frame_idx = max(image_filenames.keys())
    else:
        min_frame_idx = int(detections[:, 0].min())
        max_frame_idx = int(detections[:, 0].max())

    info_filename = os.path.join(sequence_dir, "seqinfo.ini") # 该检测序列信息ini
    if os.path.exists(info_filename):
        with open(info_filename, "r") as f:
            line_splits = [l.split('=') for l in f.read().splitlines()[1:]]
            info_dict = dict(s for s in line_splits if isinstance(s, list) and len(s) == 2)

        update_ms = 1000 / int(info_dict["frameRate"])
    else:
        update_ms = None

    feature_dim = detections.shape[1] - 10 if detections is not None else 0 # 这里的appearence大小是138，减去10以后为128，其中10位det格式的
    seq_info = {
        "sequence_name": os.path.basename(sequence_dir),
        "image_filenames": image_filenames,
        "detections": detections,
        "groundtruth": groundtruth,
        "image_size": image_size,
        "min_frame_idx": min_frame_idx,
        "max_frame_idx": max_frame_idx,
        "feature_dim": feature_dim,
        "update_ms": update_ms
    }
    return seq_info


def create_detections(detection_mat, frame_idx, min_height=0):
    """Create detections for given frame index from the raw detection matrix.
    对于给定的帧index创建检测结果

    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    frame_idx : int
        The frame index.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.

    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.

    """
    frame_indices = detection_mat[:, 0].astype(np.int) # 对应的检测帧indices
    mask = frame_indices == frame_idx # 获取当前帧的mask

    detection_list = []
    # 对于正确的mask，比如第一帧，获取所有的检测框和对应的appearance
    for row in detection_mat[mask]:
        # 其中npy的结构是framd_id，-1，bbox_x，bbox_y，bbox_w，bbox_h，bbox_score，appearance
        bbox, confidence, feature = row[2:6], row[6], row[10:]
        # 过滤height过小的bbox
        if bbox[3] < min_height:
            continue
        # 检测列表增加每一个过滤后的bbox，confidence和feature
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list


def run(sequence_dir, detection_file, output_file, min_confidence,
        nms_max_overlap, min_detection_height, max_cosine_distance,
        nn_budget, display):
    """Run multi-target tracker on a particular sequence.
    特定序列下的多目标跟踪器

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    detection_file : str
        Path to the detections file.
    output_file : str
        Path to the tracking output file. This file will contain the tracking
        results on completion.
    min_confidence : float
        Detection confidence threshold. Disregard all detections that have
        a confidence lower than this value.
    nms_max_overlap: float
        Maximum detection overlap (non-maxima suppression threshold).
    min_detection_height : int
        Detection height threshold. Disregard all detections that have
        a height lower than this value.
    max_cosine_distance : float
        Gating threshold for cosine distance metric (object appearance).
        最大cosine距离，对于cosine距离度量门控阈值，目标appearance
    nn_budget : Optional[int]
        Maximum size of the appearance descriptor gallery. If None, no budget
        is enforced.
        appearance描述子最大大小
    display : bool
        If True, show visualization of intermediate tracking results.

    """
    seq_info = gather_sequence_info(sequence_dir, detection_file) # 获得当前序列的所有信息
    metric = nn_matching.NearestNeighborDistanceMetric(
        "cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric) # 跟踪器，输入为相应的跟踪度量metric
    results = []

    def frame_callback(vis, frame_idx):
        # 每一帧回调
        print("Processing frame %05d" % frame_idx)

        # Load image and generate detections.
        # 加载图像并且生成检测结果
        detections = create_detections(
            seq_info["detections"], frame_idx, min_detection_height)
        # 过滤置信度过低的检测，仅保留高检测置信度
        detections = [d for d in detections if d.confidence >= min_confidence]

        # Run non-maxima suppression.
        # 运行NMS非极大值抑制
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # 非极大值抑制针对boxes，overlap和scores对于遮挡较大的boxes进行抑制
        indices = preprocessing.non_max_suppression(
            boxes, nms_max_overlap, scores)
        # 保留抑制后的detections
        detections = [detections[i] for i in indices]

        # Update tracker.
        # 更新跟踪
        tracker.predict()
        tracker.update(detections)

        # Update visualization.
        if display:
            image = cv2.imread(
                seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)
            vis.set_image(image.copy())
            vis.draw_detections(detections)
            vis.draw_trackers(tracker.tracks)

        # Store results.
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlwh()
            results.append([
                frame_idx, track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])

    # Run tracker.
    # 运行跟踪器
    if display:
        # 是否可视化结果，可视化间隔为5ms
        visualizer = visualization.Visualization(seq_info, update_ms=5)
    else:
        visualizer = visualization.NoVisualization(seq_info)
    # 可视化控件frame回调函数
    visualizer.run(frame_callback)

    # Store results.
    f = open(output_file, 'w')
    for row in results:
        print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (
            row[0], row[1], row[2], row[3], row[4], row[5]),file=f)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Deep SORT")
    # sequence_dir跟踪数据集MOT目录
    parser.add_argument(
        "--sequence_dir", help="Path to MOTChallenge sequence directory",
        default=None, required=True)
    # 自定义检测文件
    parser.add_argument(
        "--detection_file", help="Path to custom detections.", default=None,
        required=True)
    # 跟踪输出路径
    parser.add_argument(
        "--output_file", help="Path to the tracking output file. This file will"
        " contain the tracking results on completion.",
        default="/tmp/hypotheses.txt")
    # 检测最小置信度
    parser.add_argument(
        "--min_confidence", help="Detection confidence threshold. Disregard "
        "all detections that have a confidence lower than this value.",
        default=0.8, type=float)
    # 检测最小的bbbox高度
    parser.add_argument(
        "--min_detection_height", help="Threshold on the detection bounding "
        "box height. Detections with height smaller than this value are "
        "disregarded", default=0, type=int)
    # 非极大值最大overlap
    parser.add_argument(
        "--nms_max_overlap",  help="Non-maxima suppression threshold: Maximum "
        "detection overlap.", default=1.0, type=float)
    # 最大cosine距离
    parser.add_argument(
        "--max_cosine_distance", help="Gating threshold for cosine distance "
        "metric (object appearance).", type=float, default=0.2)
    # appearance描述子最大大小
    parser.add_argument(
        "--nn_budget", help="Maximum size of the appearance descriptors "
        "gallery. If None, no budget is enforced.", type=int, default=None)
    # 是否展示跟踪结果
    parser.add_argument(
        "--display", help="Show intermediate tracking results",
        default=True, type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.detection_file, args.output_file,
        args.min_confidence, args.nms_max_overlap, args.min_detection_height,
        args.max_cosine_distance, args.nn_budget, args.display)
