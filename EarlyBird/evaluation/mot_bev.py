import motmetrics as mm
import numpy as np


def mot_metrics_pedestrian(tSource, gtSource):
    gt = np.loadtxt(gtSource, delimiter=',')
    t = np.loadtxt(tSource, delimiter=',')

    acc = mm.MOTAccumulator()
    for frame in np.unique(gt[:, 0]).astype(int):
        gt_dets = gt[gt[:, 0] == frame][:, (1, 7, 8)]
        t_dets = t[t[:, 0] == frame][:, (1, 7, 8)]

        C = mm.distances.norm2squared_matrix(gt_dets[:, 1:3]  * 0.025 , t_dets[:, 1:3]  * 0.025, max_d2=1)  # format: gt, t
        C = np.sqrt(C)

        acc.update(gt_dets[:, 0].astype('int').tolist(),
                   t_dets[:, 0].astype('int').tolist(),
                   C,
                   frameid=frame)

    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=mm.metrics.motchallenge_metrics)
    return summary
