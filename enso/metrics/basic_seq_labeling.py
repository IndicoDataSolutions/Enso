"""Basic sequence labeling metrics."""
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score
from enso.metrics import SequenceLabelingMetric
from enso.registry import Registry, ModeKeys


def overlap(x1, x2):
    return x1["end"] >= x2["start"] and x2["end"] >= x1["start"]


def label_chars(sequence, cls, null_label="none"):
    output = []
    for item in sequence:
        if item["label"] == cls:
            output.extend([null_label] * (item["start"] - len(output)))
            output.extend([item["label"]] * (item["end"] - item["start"]))
    return output

def tp_fp_fn(true_char_labels, pred_char_labels, target_class):
    tp = 0
    fp = 0
    fn = 0
    for true, pred in zip(true_char_labels, pred_char_labels):
        if target_class == true and true == pred:
            tp += 1
        elif target_class == pred and pred != true:
            fp += 1
        elif target_class == true and true != pred:
            fn += 1
    return tp, fp, fn

def tf_fp_fn_all_classes(truth, result, none_label="none"):
    cls = set([x["label"] for y in truth + result for x in y])
    output_metrics = {}
    for c in cls:
        if c == none_label:
            continue
        true_char_labels, pred_char_labels = convert_to_per_char_labs(truth, result, c, null_label=none_label)
        output_metrics[c] = tp_fp_fn(true_char_labels, pred_char_labels, c)
    return output_metrics

def convert_to_per_char_labs(truth, result, cls, null_label="none"):
    true_out = []
    res_out = []
    for truth_b, res_b in zip(truth, result):
        truth_per_char = label_chars(truth_b, cls, null_label)
        res_per_char = label_chars(res_b, cls, null_label)
        res_per_char += [null_label] * (len(truth_per_char) - len(res_per_char))
        res_per_char = res_per_char[:len(truth_per_char)]
        true_out.extend(truth_per_char)
        res_out.extend(res_per_char)
    return true_out, res_out


@Registry.register_metric(ModeKeys.SEQUENCE)
class MicroCharPrecision(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        result_dict = tf_fp_fn_all_classes(ground_truth, result)
        tp, fp, fn = 0, 0, 0
        for tp_i, fp_i, fn_i in result_dict.values():
            tp += tp_i
            fp += fp_i
            fn += fn_i
        if tp + fp == 0:
            return 0.0
        return tp/float(tp+fp)

@Registry.register_metric(ModeKeys.SEQUENCE)
class MicroCharRecall(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        result_dict = tf_fp_fn_all_classes(ground_truth, result)
        tp, fp, fn = 0, 0, 0
        for tp_i, fp_i, fn_i in result_dict.values():
            tp += tp_i
            fp += fp_i
            fn += fn_i
        if fn + tp == 0:
            return 0.0
        return tp / float(fn + tp)



@Registry.register_metric(ModeKeys.SEQUENCE)
class MicroCharF1(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        result_dict = tf_fp_fn_all_classes(ground_truth, result)
        tp, fp, fn = 0, 0, 0
        for tp_i, fp_i, fn_i in result_dict.values():
            tp += tp_i
            fp += fp_i
            fn += fn_i
        if fn + tp == 0 or tp+fp == 0:
            return 0.0
        recall = tp / float(fn + tp)
        precision = tp/float(tp+fp)
        if precision + recall == 0.0:
            return 0.0
        return 2 * (recall * precision) / (recall + precision)

@Registry.register_metric(ModeKeys.SEQUENCE)
class MacroCharF1(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        f1s = []
        
        result_dict = tf_fp_fn_all_classes(ground_truth, result)
        for tp, fp, fn in result_dict.values():
            if fn + tp == 0 or tp+fp == 0:
                f1s.append(0.0)
            else:
                recall = tp / float(fn + tp)
                precision = tp/float(tp+fp)
                if precision + recall == 0.0:
                    f1s.append(0.0)
                else:
                    f1s.append(2 * (recall * precision) / (recall + precision))
        return sum(f1s)/len(f1s)

@Registry.register_metric(ModeKeys.SEQUENCE)
class MacroCharRecall(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        recall = []

        result_dict = tf_fp_fn_all_classes(ground_truth, result)
        for tp, fp, fn in result_dict.values():
            if fn + tp == 0:
                recall.append(0.0)
            else:
                recall.append(tp / float(fn + tp))
            
        return sum(recall)/len(recall)

@Registry.register_metric(ModeKeys.SEQUENCE)
class MacroCharPrecision(SequenceLabelingMetric):
    """ Accuracy of overlaps """

    def evaluate(self, ground_truth, result):
        precs = []
        result_dict = tf_fp_fn_all_classes(ground_truth, result)
        for tp, fp, fn in result_dict.values():
            if tp+fp == 0:
                precs.append(0.0)
            else:
                precs.append(tp/float(tp+fp))
        return sum(precs)/len(precs)
