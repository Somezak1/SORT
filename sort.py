"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
import os
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

np.random.seed(0)


def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])  #
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2],  (m,5)  (n,5)
    """
    bb_gt = np.expand_dims(bb_gt, 0)             # (1,n,5)
    bb_test = np.expand_dims(bb_test, 1)         # (m,1,5)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return (o)  # (m, n)


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
      [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
      the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if (score == None):
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        self.kf.F = np.array(  # 状态转移方程 (7, 7)
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        # x = x + x*
        # y = y + y*
        # s = s + s*
        # r = r
        # x* = x*
        # y* = y*
        # s* = s*

        self.kf.H = np.array(  # 量测方程 (4, 7)
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        # x
        # y
        # s
        # r

        self.kf.R[2:, 2:] *= 10.  # 量测噪声 eye(4, 4)
        # array([[1., 0., 0., 0.],
        #        [0., 1., 0., 0.],
        #        [0., 0., 10., 0.],
        #        [0., 0., 0., 10.]])

        self.kf.P[4:, 4:] *= 1000.  # give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.  # 协方差矩阵 eye(7, 7) ，这个会随predict和update而变化
        # array([[10., 0., 0., 0., 0., 0., 0.],
        #        [0., 10., 0., 0., 0., 0., 0.],
        #        [0., 0., 10., 0., 0., 0., 0.],
        #        [0., 0., 0., 10., 0., 0., 0.],
        #        [0., 0., 0., 0., 10000., 0., 0.],
        #        [0., 0., 0., 0., 0., 10000., 0.],
        #        [0., 0., 0., 0., 0., 0., 10000.]])

        self.kf.Q[-1, -1] *= 0.01  # 过程噪声（系统噪声） eye(7, 7)
        self.kf.Q[4:, 4:] *= 0.01
        # array([[1.    , 0.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
        #        [0.    , 1.    , 0.    , 0.    , 0.    , 0.    , 0.    ],
        #        [0.    , 0.    , 1.    , 0.    , 0.    , 0.    , 0.    ],
        #        [0.    , 0.    , 0.    , 1.    , 0.    , 0.    , 0.    ],
        #        [0.    , 0.    , 0.    , 0.    , 1e-02 , 0.    , 0.    ],
        #        [0.    , 0.    , 0.    , 0.    , 0.    , 1e-02 , 0.    ],
        #        [0.    , 0.    , 0.    , 0.    , 0.    , 0.    , 1e-04]])

        self.kf.x[:4] = convert_bbox_to_z(bbox)  # [x, y, s, r, 0, 0, 0]
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0         # 没用到
        self.hit_streak = 0
        self.age = 0          # 没用到

    def update(self, bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0   # 一旦该对象在当前帧有框与之匹配，则这个值就会清零
        self.history = []
        self.hits += 1               # 没用到
        self.hit_streak += 1         # 该目标在连续hit_streak个帧里都得到了追踪
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if ((self.kf.x[6] + self.kf.x[2]) <= 0):  # s+s* <= 0  s肯定大于0，因此若s+s* <= 0 ，必然是s为负的太多
            self.kf.x[6] *= 0.0                   # s* = 0
        self.kf.predict()
        self.age += 1                     # 没用到
        if (self.time_since_update > 0):  # 初始为0
            self.hit_streak = 0
        self.time_since_update += 1       # 只要该对象在当前帧没有框与之匹配，则这个值就会一直增加
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to tracked object (both represented as bounding boxes)

    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    """
    if (len(trackers) == 0):
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)  # (m,5)  (n,5) 得到 (m,n)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:     # a.sum(1)  当前帧每个预测框与上一帧检测框对应预测框IOU大于阈值的个数
            matched_indices = np.stack(np.where(a), axis=1) # a.sum(0)  上一帧每个检测框对应预测框与当前帧预测框IOU大于阈值的个数
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))  # detections 形状为 (0, 5)
    # matched_indices[i][j]  表明当前帧第i个预测框与上一帧第j个检测框在当前帧的预测框匹配

    unmatched_detections = []
    for d, det in enumerate(detections):
        if (d not in matched_indices[:, 0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if (t not in matched_indices[:, 1]):
            unmatched_trackers.append(t)

    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if (iou_matrix[m[0], m[1]] < iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if (len(matches) == 0):
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []  # 目前仍在跟踪的目标
        self.frame_count = 0

    def update(self, dets=np.empty((0, 5))):
        """
        Params:
          dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]  # 根据上一帧中每个检测框的坐标，预测该检测框在下一帧中的坐标
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))  # 去除形式不正常的检测框
        for t in reversed(to_del):  # trackers 为 上一帧中每个检测框的坐标
            self.trackers.pop(t)    # trks 为 根据上一帧中每个检测框的坐标预测得到的下一帧该检测框的坐标
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks,
                                                                                   self.iou_threshold)  # iou_threshold=0.3
        len_dets = len(dets)
        len_trks = len(trks)
        len_mat = len(matched)
        len_unmat_dets = len(unmatched_dets)
        len_unmat_trks = len(unmatched_trks)
        assert len_mat + len_unmat_dets == len_dets and len_mat + len_unmat_trks == len_trks
        # 第一次时
        # matched: np.empty((0, 2))
        # unmatched_dets: np.arange(len(dets))
        # unmatched_trks: np.empty((0, 5))

        # 之后
        # matched: (a, 2)
        # unmatched_dets: (b, )
        # unmatched_trks: (c, )

        # update matched trackers with assigned detections
        for m in matched:  # m表明当前帧第i个预测框与上一帧第j个检测框在当前帧的预测框匹配
            self.trackers[m[1]].update(dets[m[0], :])  # 使用与上一帧第j个检测框在当前帧的预测框匹配的当前帧第i个预测框来对
                                                       # 上一帧第j个检测框在当前帧的预测框进行修正

        # create and initialise new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])  # dets[i,:], (5,)  [x1,y1,x2,y2,score]
            self.trackers.append(trk)
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]  # (x1, y1, x2, y2)
            if (trk.time_since_update < 1) and (  # (trk.time_since_update < 1)   当前帧新增的trackers和这一帧得到更新的旧trackers会满足这个条件
                    trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):  # min_hits=3
                    # self.frame_count <= self.min_hits  前三帧中，每帧新增的trackers和得到匹配的trackers会被显示/ret
                    # 从第四帧开始，需要是至少连续三帧得到匹配的trackers才会被显示/ret
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))  # +1 as MOT benchmark requires positive
            i -= 1
            pop_times = 0
            # remove dead tracklet
            if (trk.time_since_update > self.max_age):  # max_age=1，连续两帧没有得到匹配的trackers会被剔除
                self.trackers.pop(i)
                pop_times += 1
        if (len(ret) > 0):
            ret = np.concatenate(ret)
            return len_trks, len_dets, len_mat, len_unmat_trks, len_unmat_dets, len(ret), pop_times, ret
        return len_trks, len_dets, len_mat, len_unmat_trks, len_unmat_dets, 0, pop_times, np.empty((0, 5))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]', default=True)
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='MOT15')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # all train
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # used only for display
    if (display):
        if not os.path.exists('MOT15'):
            print(
                '\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    if not os.path.exists('output'):
        os.makedirs('output')
    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt')  # 'data\\train\\*\\det\\det.txt'
    for seq_dets_fn in glob.glob(pattern)[::-1]:           # 遍历每个视频的检测结果
        mot_tracker = Sort(max_age=args.max_age,     # 1
                           min_hits=args.min_hits,   # 3
                           iou_threshold=args.iou_threshold)  # create instance of the SORT tracker
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]  # *部分对应的内容

        with open(os.path.join('output', '%s.txt' % (seq)), 'w') as out_file:  # 将结果写入 output\\*部分对应的内容.txt 文件中去
            print("Processing %s." % (seq))
            pre_trackers = []
            pre_len_trks, pre_len_dets, pre_len_mat, pre_len_unmat_trks, pre_len_unmat_dets, pre_len_ret, pre_pop_times = 0, 0, 0, 0, 0, 0, 0
            for frame in range(int(seq_dets[:, 0].max())):   # seq_dets第一列是帧序号
                frame += 1  # detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0] == frame, 2:7]  # 这一帧图像的检测结果[x1,y1,w,h,score],如果没取到就是array([], shape=(0, 5))
                dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                start_time = time.time()
                cur_len_trks, cur_len_dets, cur_len_mat, cur_len_unmat_trks, cur_len_unmat_dets, cur_len_ret, cur_pop_times, trackers = mot_tracker.update(dets)  # (n, 5)
                cycle_time = time.time() - start_time
                total_time += cycle_time

                if (display):
                    fn = os.path.join('MOT15', phase, seq, 'img1', '%06d.jpg' % (frame))
                    im = io.imread(fn)
                    ax[1].imshow(im)
                    ax[1].set_title(f'{frame}  Trk:{cur_len_trks}  Det:{cur_len_dets}  Mat:{cur_len_mat}  UMTrks:{cur_len_unmat_trks}  UMDets:{cur_len_unmat_dets}  Disp:{cur_len_ret}  Pop:{cur_pop_times}')

                    if frame > 1:
                        fn = os.path.join('MOT15', phase, seq, 'img1', '%06d.jpg' % (frame - 1))
                        im = io.imread(fn)
                        ax[0].imshow(im)
                        ax[0].set_title(f'{frame - 1}  Trk:{pre_len_trks}  Det:{pre_len_dets}  Mat:{pre_len_mat}  UMT:{pre_len_unmat_trks}  UMD:{pre_len_unmat_dets}  Disp:{pre_len_ret}  Pop:{pre_pop_times}')

                for d in trackers:
                    # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                    #       file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax[1].add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))

                for d in pre_trackers:
                    # print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                    #       file=out_file)
                    if (display):
                        d = d.astype(np.int32)
                        ax[0].add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3, ec=colours[d[4] % 32, :]))
                pre_trackers = trackers
                pre_len_trks = cur_len_trks
                pre_len_dets = cur_len_dets
                pre_len_mat = cur_len_mat
                pre_len_unmat_trks = cur_len_unmat_trks
                pre_len_ret = cur_len_ret
                pre_len_unmat_dets = cur_len_unmat_dets
                pre_pop_times = cur_pop_times
                if (display):
                    time.sleep(1)
                    fig.canvas.flush_events()
                    plt.draw()
                    ax[0].cla()
                    ax[1].cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if (display):
        print("Note: to get real runtime results run without the option: --display")
