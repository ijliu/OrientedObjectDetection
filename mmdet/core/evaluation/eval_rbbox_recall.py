from multiprocessing import get_context
import numpy as np

from .bbox_overlaps import bbox_overlaps

from mmcv.utils import print_log
from terminaltables import AsciiTable


def _recalls(all_ious, proposal_nums, thrs):

    img_num = all_ious.shape[0]
    total_gt_num = sum([ious.shape[0] for ious in all_ious])

    _ious = np.zeros((proposal_nums.size, total_gt_num), dtype=np.float32)
    for k, proposal_num in enumerate(proposal_nums):
        tmp_ious = np.zeros(0)
        for i in range(img_num):
            ious = all_ious[i][:, :proposal_num].copy()
            gt_ious = np.zeros((ious.shape[0]))
            if ious.size == 0:
                tmp_ious = np.hstack((tmp_ious, gt_ious))
                continue
            for j in range(ious.shape[0]):
                gt_max_overlaps = ious.argmax(axis=1)
                max_ious = ious[np.arange(0, ious.shape[0]), gt_max_overlaps]
                gt_idx = max_ious.argmax()
                gt_ious[j] = max_ious[gt_idx]
                box_idx = gt_max_overlaps[gt_idx]
                ious[gt_idx, :] = -1
                ious[:, box_idx] = -1
            tmp_ious = np.hstack((tmp_ious, gt_ious))
        _ious[k, :] = tmp_ious

    _ious = np.fliplr(np.sort(_ious, axis=1))
    recalls = np.zeros((proposal_nums.size, thrs.size))
    for i, thr in enumerate(thrs):
        recalls[:, i] = (_ious >= thr).sum(axis=1) / float(total_gt_num)

    return recalls


def xywh2xyxy(bbox):
    new_bboxes = np.zeros_like(bbox)
    new_bboxes[..., 0] = bbox[..., 0] - bbox[..., 2]/2
    new_bboxes[..., 1] = bbox[..., 1] - bbox[..., 3]/2
    new_bboxes[..., 2] = bbox[..., 0] + bbox[..., 2]/2
    new_bboxes[..., 3] = bbox[..., 1] + bbox[..., 3]/2
    return new_bboxes

def eval_rbbox_recall(det_results,
                   annotations,
                   scale_ranges=None,
                   iou_thrs=0.5,
                   use_07_metric=True,
                   dataset=None,
                   logger=None,
                   nproc=4):
    assert len(det_results) == len(annotations)
    num_imgs = len(det_results)
    proposal_nums = np.array([300, 1000, 2000])
    iou_thrs = np.array([iou_thrs])

    gt_bboxes = []
    for ann in annotations:
        bboxes = ann['bboxes']
        gt_bboxes.append(bboxes)
    
    all_ious = []
    for i in range(num_imgs):
        img_proposal = det_results[i][..., :4]

        img_proposal = xywh2xyxy(img_proposal)
        gts = xywh2xyxy(gt_bboxes[i][:,:4])

        prop_num = min(img_proposal.shape[0], proposal_nums[-1])

        if gts is None or gts.shape[0] == 0:
            ious = np.zeros((0, img_proposal.shape[0]), dtype=np.float32)
        else:
            ious = bbox_overlaps(gts, img_proposal[:prop_num, :4])
        all_ious.append(ious)
    all_ious = np.array(all_ious)
    recalls = _recalls(all_ious, proposal_nums, iou_thrs)

    print_recall_summary(recalls, proposal_nums, iou_thrs, logger=logger)
    return recalls



def print_recall_summary(recalls,
                         proposal_nums,
                         iou_thrs,
                         row_idxs=None,
                         col_idxs=None,
                         logger=None):
    """Print recalls in a table.

    Args:
        recalls (ndarray): calculated from `bbox_recalls`
        proposal_nums (ndarray or list): top N proposals
        iou_thrs (ndarray or list): iou thresholds
        row_idxs (ndarray): which rows(proposal nums) to print
        col_idxs (ndarray): which cols(iou thresholds) to print
        logger (logging.Logger | str | None): The way to print the recall
            summary. See `mmdet.utils.print_log()` for details. Default: None.
    """
    proposal_nums = np.array(proposal_nums, dtype=np.int32)
    iou_thrs = np.array(iou_thrs)
    if row_idxs is None:
        row_idxs = np.arange(proposal_nums.size)
    if col_idxs is None:
        col_idxs = np.arange(iou_thrs.size)
    row_header = [''] + iou_thrs[col_idxs].tolist()
    table_data = [row_header]
    for i, num in enumerate(proposal_nums[row_idxs]):
        row = [f'{val:.3f}' for val in recalls[row_idxs[i], col_idxs].tolist()]
        row.insert(0, num)
        table_data.append(row)
    table = AsciiTable(table_data)
    print_log('\n' + table.table, logger=logger)
