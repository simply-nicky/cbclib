from typing import Dict
import numpy as np

class LSD:
    """LSD  is a class for performing the streak detection on digital images with Line Segment
    Detector algorithm [LSD]_.

    Args:
        scale : When different from 1.0, LSD will scale the input image by 'scale' factor
            by Gaussian filtering, before detecting line segments.
        sigma_scale : When ``scale`` is different from 1.0, the sigma of the Gaussian
            filter is :code:`sigma = sigma_scale / scale`, if scale is less than 1.0, and
            :code:`sigma = sigma_scale` otherwise.
        log_eps : Detection threshold, accept if -log10(NFA) > log_eps. The larger the
            value, the more strict the detector is, and will result in less detections.
            The value -log10(NFA) is equivalent but more intuitive than NFA:

            * -1.0 gives an average of 10 false detections on noise.
            * 0.0 gives an average of 1 false detections on noise.
            * 1.0 gives an average of 0.1 false detections on nose.
            * 2.0 gives an average of 0.01 false detections on noise.

        ang_th : Gradient angle tolerance in the region growing algorithm, in degrees.
        density_th : Minimal proportion of 'supporting' points in a rectangle.
        quant : Bound to the quantization error on the gradient norm. Example: if gray
            levels are quantized to integer steps, the gradient (computed by finite
            differences) error due to quantization will be bounded by 2.0, as the worst
            case is when the error are 1 and -1, that gives an error of 2.0.

    References:
        .. [LSD] "LSD: a Line Segment Detector" by Rafael Grompone von Gioi, Jeremie Jakubowicz,
                Jean-Michel Morel, and Gregory Randall, Image Processing On Line, 2012,
                DOI: 10.5201/ipol.2012.gjmr-lsd, http://dx.doi.org/10.5201/ipol.2012.gjmr-lsd.
    """
    scale       : float
    sigma_scale : float
    log_eps     : float
    ang_th      : float
    quant       : float

    def __init__(self, scale: float=0.8, sigma_scale: float=0.6, log_eps: float=0.0,
                 ang_th: float=45.0, density_th: float=0.7, quant: float=2.0) -> None:
        ...

    def detect(self, image: np.ndarray, cutoff: float, filter_threshold: float=0.0,
               group_threshold: float=0.6, filter: bool=True, group: bool=True,
               dilation: float=0.0, return_labels: bool=False,
               num_threads: int=1) -> Dict[str, Dict[int, np.ndarray]]:
        """Perform the LSD streak detection on an input array `image`. The Streak detection
        comprises three steps: an initial LSD detection of lines, a grouping of the detected
        lines and merging, if the normalized cross-correlation value if higher than the
        ``group_threshold``, discarding the lines with a 0-order image moment lower than
        ``filter_threshold``.

        Args:
            image : 2D array of the digital image.
            cutoff : Distance cut-off value for lines grouping in pixels.
            filter_threshold : Filtering threshold. A line is discarded if the 0-order image
                moment is lower than ``filter_threshold``. 
            group_threshold : Grouping threshold. The lines are merged if the cross-correlation
                value of a pair of lines is higher than ``group_threshold``.
            filter : Perform filtering if True.
            group : Perform grouping if True.
            dilation : Line mask dilation value in pixels.
            return_labels : Return line labels mask if True.
            num_threads : A number of threads used in the computations.

        Returns:
            A dictionary with detection results. The dictionary contains a list of the following
            attributes:

            * `lines` : An array of the detected lines. Each line is comprised of 7 parameters
              as follows:

              * `[x1, y1]`, `[x2, y2]` : The coordinates of the line's ends.
              * `width` : Line's width.
              * `p` : Angle precision [0, 1] given by angle tolerance over 180 degree.
              * `-log10(NFA)` : Number of false alarms.

            * `labels` : Image where each pixel indicates the line segment to which it belongs.
              Unused pixels have the value 0, while the used ones have the number of the line
              segment, numbered in the same order as in `lines`.
        """
        ...
