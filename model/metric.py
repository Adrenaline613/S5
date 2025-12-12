import numpy as np
import torch
from scipy.stats import hmean
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix
import prettytable


def samples_metrics(num_list, print_table=True):
    """
    Calculate and optionally print classification metrics based on the number of true negatives (TN), false positives (FP),
    false negatives (FN), and true positives (TP).

    Args:
        num_list (list): A list containing the number of TN, FP, FN, and TP.
        print_table (bool): Whether to print the metrics in a table format. Default is True.

    Returns:
        tuple: A tuple containing accuracy, precision, recall, specificity, and F1 score.
    """

    tn, fp, fn, tp = num_list[0], num_list[1], num_list[2], num_list[3]

    acc = np.round(((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) != 0 else 0) * 100, 2)
    pre = np.round((tp / (tp + fp) if (tp + fp) != 0 else 0) * 100, 2)
    sen = np.round((tp / (tp + fn) if (tp + fn) != 0 else 0) * 100, 2)
    spe = np.round((tn / (tn + fp) if (tn + fp) != 0 else 0) * 100, 2)
    f1 = np.round((2 * (pre * sen) / (pre + sen) if (pre + sen) != 0 else 0), 2)

    if print_table:

        tabel_metrics = prettytable.PrettyTable(['ACC(%)', 'PRE(%)', 'REC(%)', 'SPE(%)', 'F1(%)', '   ', 'CM', 'Pred 0', 'Pred 1'])
        tabel_metrics.add_row([acc, pre, sen, spe, f1, ' ', 'True 0', tn, fp])
        tabel_metrics.add_row([' ', ' ', ' ', ' ', ' ', ' ', 'True 1', fn, tp])
        tabel_metrics.title = 'Samples Metrics'

        print(tabel_metrics)

    return acc, pre, sen, spe, f1


def get_samples_metric_num_batch(pred_vector_batch, true_vector_batch):
    """
    Calculate the confusion matrix components (TN, FP, FN, TP) for a batch of predicted and true labels.

    Args:
        pred_vector_batch (numpy.ndarray or torch.Tensor): Batch of predicted labels.
        true_vector_batch (numpy.ndarray or torch.Tensor): Batch of true labels.

    Returns:
        list: A list containing the number of true negatives (TN), false positives (FP), false negatives (FN), and true positives (TP).
    """

    if isinstance(pred_vector_batch, torch.Tensor):
        pred_vector_batch = pred_vector_batch.cpu().numpy()
    if isinstance(true_vector_batch, torch.Tensor):
        true_vector_batch = true_vector_batch.cpu().numpy()

    if len(pred_vector_batch.shape) != 1:
        pred_vector_batch = pred_vector_batch.flatten()
    if len(true_vector_batch.shape) != 1:
        true_vector_batch = true_vector_batch.flatten()

    cm = confusion_matrix(true_vector_batch, pred_vector_batch, labels=[0, 1])
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    return [tn, fp, fn, tp]


def overlap_metrics(num_list, overlap_thresholds, print_threshold=0.2, print_table=True):
    """
    Calculate and optionally print overlap metrics based on the number of detected spindles, ground truth spindles,
    true positives, and IoU sum for each overlap threshold.

    Args:
        num_list (list): A list containing the number of detected spindles, ground truth spindles, true positives,
                         and IoU sum for each overlap threshold.
        overlap_thresholds (numpy.ndarray): Array of overlap thresholds.
        print_threshold (float): The overlap threshold to print metrics for. Default is 0.2.
        print_table (bool): Whether to print the metrics in a table format. Default is True.

    Returns:
        tuple: A tuple containing precision, recall, F1 score, mean F1 score, and mean IoU at the specified print threshold.
    """

    n_detected_spindles, n_gs_spindles, n_tp, all_iou_sum = num_list[0], num_list[1], num_list[2], num_list[3]

    if (n_detected_spindles == 0) & (n_gs_spindles == 0):
        # If there are no spindles detected and the gold standard doesn't contain any spindles, the precision, recall
        # and f1 score are defined as one
        precision = np.ones_like(n_tp)
        recall = np.ones_like(n_tp)
        f1 = np.ones_like(n_tp)
    elif (n_detected_spindles == 0) | (n_gs_spindles == 0):
        # If either there are no spindles detected or the gold standard doesn't contain any spindles, there can't be any
        # true positives and precision, recall and f1 score are defined as zero
        precision = np.zeros_like(n_tp)
        recall = np.zeros_like(n_tp)
        f1 = np.zeros_like(n_tp)
    else:
        # Precision is defined as TP/(TP+FP)
        precision = n_tp / n_detected_spindles
        # Recall is defined as TP/(TP+FN)
        recall = n_tp / n_gs_spindles
        # f1 score is defined as harmonic mean between precision and recall
        f1 = hmean(np.c_[precision, recall], axis=1)

    mf1 = np.mean(f1)
    miou = np.array([iou / tp if tp != 0 else 0 for iou, tp in zip(all_iou_sum, n_tp)])

    idx = np.where(overlap_thresholds == print_threshold)[0][0]
    print_pre = np.round(precision[idx] * 100, 2)
    print_rec = np.round(recall[idx] * 100, 2)
    print_f1 = np.round(f1[idx] * 100, 2)
    print_mf1 = np.round(mf1 * 100, 2)
    print_miou = np.round(miou[idx] * 100, 2)

    if print_table:

        tabel_metrics = prettytable.PrettyTable(['PRE(%)', 'REC(%)', 'F1(%)', 'mF1(%)', 'mIoU(%)'])
        tabel_metrics.add_row([print_pre, print_rec, print_f1, print_mf1, print_miou])
        tabel_metrics.title = f'Overlap Metrics (threshold = {print_threshold})'
        print(tabel_metrics)

    return print_pre, print_rec, print_f1, print_mf1, print_miou


def get_overlap_metric_num_batch(pred_vector_batch, true_vector_batch, overlap_thresholds):
    """
    https://github.com/dslaborg/sumo
    Compute overlap metrics for a batch of predicted and true spindle vectors.

    Args:
        pred_vector_batch (numpy.ndarray or torch.Tensor): Batch of predicted spindle vectors.
        true_vector_batch (numpy.ndarray or torch.Tensor): Batch of true spindle vectors.
        overlap_thresholds (numpy.ndarray): Array of overlap thresholds for computing true positives.

    Returns:
        tuple: Number of predicted spindles, number of true spindles, number of true positives, and sum of IoUs.
    """

    # Convert tensors to numpy arrays if they are tensors
    if isinstance(pred_vector_batch, torch.Tensor):
        pred_vector_batch = pred_vector_batch.cpu().numpy()
    if isinstance(true_vector_batch, torch.Tensor):
        true_vector_batch = true_vector_batch.cpu().numpy()

    # Initialize the number of detected spindles, true spindles, true positives and batch IOU sum
    n_spindles_pred, n_spindles_true, n_tp, batch_iou_sum = (
        0, 0, np.zeros_like(overlap_thresholds, dtype=int), np.zeros_like(overlap_thresholds))

    # Iterate over each pair of predicted and true spindle vectors
    for spindles_pred, spindles_true in zip(pred_vector_batch, true_vector_batch):
        spindle_pred_indices = spindle_vect_to_indices(spindles_pred)
        spindle_true_indices = spindle_vect_to_indices(spindles_true)
        true_positives, iou_sum = get_true_positives(spindle_pred_indices, spindle_true_indices, overlap_thresholds)

        n_spindles_pred += spindle_pred_indices.shape[0]
        n_spindles_true += spindle_true_indices.shape[0]
        n_tp += true_positives
        batch_iou_sum += iou_sum

    return [n_spindles_pred, n_spindles_true, n_tp, batch_iou_sum]


def spindle_vect_to_indices(x):
    """
    https://github.com/dslaborg/sumo
    Convert a spindle vector to start and end indices of spindles.

    Args:
        x (numpy.ndarray): Spindle vector where 1 indicates the presence of a spindle and 0 indicates its absence.

    Returns:
        numpy.ndarray: Array of start and end indices of spindles.
    """

    # Calculate the difference between consecutive elements, adding 0 at both ends
    diff = np.diff(np.r_[0, x, 0])

    # Find indices where the difference is 1 (start of spindle) and -1 (end of spindle)
    start_indices = np.argwhere(diff == 1)
    end_indices = np.argwhere(diff == -1)

    # Combine start and end indices into a single array
    spindle_indices = np.c_[start_indices, end_indices]

    return spindle_indices


def get_true_positives(spindles_detected, spindles_gs, overlap_thresholds):
    """
    https://github.com/dslaborg/sumo
    Calculate the number of true positives and the sum of Intersection over Union (IoU) for detected spindles.

    Args:
        spindles_detected (numpy.ndarray): Array of start and end indices of detected spindles.
        spindles_gs (numpy.ndarray): Array of start and end indices of ground truth spindles.
        overlap_thresholds (numpy.ndarray): Array of overlap thresholds for determining true positives.

    Returns:
        tuple: Number of true positives for each threshold and the sum of IoUs for each threshold.
    """

    # If either there is no spindle detected or the gold standard doesn't contain any spindles, there can't be any true
    # positives
    if (spindles_detected.shape[0] == 0) or (spindles_gs.shape[0] == 0):
        return np.zeros_like(overlap_thresholds, dtype=np.int8), np.zeros_like(overlap_thresholds)

    # Use Hungarian algorithm to obtain a one-to-one matching that maximizes total overlap.
    # linear_sum_assignment minimizes cost, so pass -overlap to maximize overlap.
    row_ind, col_ind = linear_sum_assignment(-overlap)
    overlap_valid = np.zeros_like(overlap)
    for i, j in zip(row_ind, col_ind):
        overlap_valid[i, j] = overlap[i, j]

    iou_sum = np.empty_like(overlap_thresholds)
    n_tp = np.empty_like(overlap_thresholds, dtype=np.int8)

    # Calculate the valid matches (true positives) depending on the overlap threshold
    for idx, overlap_threshold in enumerate(overlap_thresholds):
        # The sum of the overlaps above the threshold
        iou_sum[idx] = np.sum(overlap_valid[overlap_valid > overlap_threshold])
        # All remaining values > overlap_threshold are valid matches (true positives)
        matches = np.argwhere(overlap_valid > overlap_threshold)
        n_tp[idx] = matches.shape[0]

    return n_tp, iou_sum


def get_overlap(spindles_detected, spindles_gs):
    """
    https://github.com/dslaborg/sumo
    Compute the overlap (Intersection over Union, IoU) between detected and ground truth spindles.

    Args:
        spindles_detected (numpy.ndarray): Array of start and end indices of detected spindles.
        spindles_gs (numpy.ndarray): Array of start and end indices of ground truth spindles.

    Returns:
        numpy.ndarray: Matrix of IoU values between each pair of detected and ground truth spindles.
    """

    n_detected_spindles = spindles_detected.shape[0]
    n_gs_spindles = spindles_gs.shape[0]

    # The (relative) overlap between each pair of detected spindle and gs spindle
    overlap = np.empty((n_detected_spindles, n_gs_spindles))

    for index in np.ndindex(n_detected_spindles, n_gs_spindles):
        idx_detected, idx_gs = spindles_detected[index[0]], spindles_gs[index[1]]
        # [start, stop) indices of the detected spindle and of the gs spindle
        idx_range_detected, idx_range_gs = np.arange(idx_detected[0], idx_detected[1]), np.arange(idx_gs[0], idx_gs[1])

        # Calculate intersect and union of the spindle indices
        intersect = np.intersect1d(idx_range_detected, idx_range_gs, assume_unique=True)
        union = np.union1d(idx_range_detected, idx_range_gs)

        # Overlap of a detected spindle and a gs spindle is defined as the intersect over the union
        overlap[index] = intersect.shape[0] / union.shape[0]  # type: ignore

    return overlap
