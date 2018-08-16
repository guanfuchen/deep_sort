# coding=utf-8
# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.
    这个类表示在一帧中的bbox检测，同时包括检测框，检测置信度和对应的appearance

    Parameters
    ----------
    tlwh : array_like
        Bounding box in format `(x, y, w, h)`.
    confidence : float
        Detector confidence score.
    feature : array_like
        A feature vector that describes the object contained in this image.

    Attributes
    ----------
    tlwh : ndarray
        Bounding box in format `(top left x, top left y, width, height)`.
    confidence : ndarray
        Detector confidence score.
    feature : ndarray | NoneType
        A feature vector that describes the object contained in this image.

    """

    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=np.float) # 对应的tlwh
        self.confidence = float(confidence) # 对应的置信度
        self.feature = np.asarray(feature, dtype=np.float32) # 对应的特征

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        将对应的bbox tlwh转换为xyah
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2 # 将tl转换为中心坐标的c_x和c_y
        ret[2] /= ret[3]
        return ret
