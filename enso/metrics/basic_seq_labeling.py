"""Basic sequence labeling metrics."""
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score
from enso.metrics import SequenceLabelingMetric
from enso.registry import Registry, ModeKeys


def overlap(x1, x2):
    return x1["end"] >= x2["start"] and x2["end"] >= x1["start"]


def label_chars(sequence, null_label="none"):
    output = []
    for item in sequence:
        output.extend([null_label] * (item["start"] - len(output)))
        output.extend([item["label"]] * (item["end"] - item["start"]))
    return output

def tp_fp_fn(true_char_labels, pred_char_labels, target_class):
    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(true_char_labels, pred_char_labels):
        print(target_class, true, pred)
        if target_class == true and true == pred:
            tp += 1
        elif target_class == pred and pred != true:
            fp += 1
        elif target_class == true and true != pred:
            fn += 1
    print(tp, fp, fn)
    return tp, fp, fn

def tf_fp_fn_all_classes(true_char_labels, pred_char_labels, none_label="none"):
    cls = set(true_char_labels + pred_char_labels)
    output_metrics = {}
    for c in cls:
        if c == none_label:
            continue
        output_metrics[c] = tp_fp_fn(true_char_labels, pred_char_labels, c)
    return output_metrics

def convert_to_per_char_labs(truth, result, null_label="none"):
    true_out = []
    res_out = []
    for truth_b, res_b in zip(truth, result):
        truth_per_char = label_chars(truth_b, null_label)
        res_per_char = label_chars(res_b, null_label)
        res_per_char += [null_label] * (len(truth_per_char) - len(res_per_char))
        res_per_char = res_per_char[:len(truth_per_char)]
        true_out.extend(truth_per_char)
        res_out.extend(res_per_char)
    return true_out, res_out


@Registry.register_metric(ModeKeys.SEQUENCE)
class MicroCharPrecision(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        result_dict = tf_fp_fn_all_classes(*convert_to_per_char_labs(ground_truth, result))
        tp, fp, fn = 0, 0, 0
        for tp_i, fp_i, fn_i in result_dict.values():
            tp += tp_i
            fp += fp_i
            fn += fn_i
        return tp/float(tp+fp)

@Registry.register_metric(ModeKeys.SEQUENCE)
class MicroCharRecall(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        result_dict = tf_fp_fn_all_classes(*convert_to_per_char_labs(ground_truth, result))
        tp, fp, fn = 0, 0, 0
        for tp_i, fp_i, fn_i in result_dict.values():
            tp += tp_i
            fp += fp_i
            fn += fn_i
        return fp / float(fn + tp)



@Registry.register_metric(ModeKeys.SEQUENCE)
class MicroCharF1(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        result_dict = tf_fp_fn_all_classes(*convert_to_per_char_labs(ground_truth, result))
        tp, fp, fn = 0, 0, 0
        for tp_i, fp_i, fn_i in result_dict.values():
            tp += tp_i
            fp += fp_i
            fn += fn_i
        recall = fp / float(fn + tp)
        precision = tp/float(tp+fp)
        return 2 * (recall * precision) / (recall + precision)