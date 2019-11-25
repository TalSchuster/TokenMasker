#from allennlp.commands.evaluate import evaluate_from_args
import argparse
import os
import sys
from itertools import product
import jsonlines
from tqdm import tqdm

from typing import Dict, Any, Iterable
import argparse
import logging
import json

import torch

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu
from allennlp.common.util import prepare_environment
from allennlp.common.tqdm import Tqdm
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.iterators import DataIterator
from allennlp.models.archival import load_archive
from allennlp.models.model import Model
from allennlp.nn import util
from allennlp.common.util import import_submodules

'''
uses the mask_generator model to output masks

'''


label_dict = {0: "SUPPORTS",
              1: "NOT ENOUGH INFO",
              2: "REFUTES"
              }

TARGET_LABEL = 1
DBG_OUTPUT_BY_DIST = False

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--cuda_device', type=int, help='id of GPU to use', default=-1)
parser.add_argument('-w', '--weights_file', type=str, help='path of weights file to use', default='')
parser.add_argument('-f', '--archive_file', type=str, help='path to tar.gz file path', required=True)
parser.add_argument('-i', '--input_file', type=str, help='path to the file containing the evaluation data', required=True)
parser.add_argument('-out', '--preds_file', type=str, help='output file to save the results', default='tmp.jsonl')
parser.add_argument('-append', action='store_true', help='allow append to previous run', default=False)
parser.add_argument('-t', '--target_label', type=int, help='label ind', default=TARGET_LABEL)

args = parser.parse_args()
import_submodules('masker_allen_pkg')


def evaluate(model: Model,
             instances: Iterable[Instance],
             data_iterator: DataIterator,
             cuda_device: int,
             preds_file: str = "") -> Dict[str, Any]:
    _warned_tqdm_ignores_underscores = False
    check_for_gpu(cuda_device)

    if preds_file:
        out_writer = jsonlines.open(preds_file, mode='a')

    with torch.no_grad():
        model.eval()

        iterator = data_iterator(instances,
                                 num_epochs=1,
                                 shuffle=False)
        print("Iterating over dataset")
        generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

        batch_count = 0
        loss_count = 0
        total_loss = 0.0
        total_corrects = 0
        total_count = 0

        tot_teacher_mask = 0
        tot_teacher_covered = 0
        tot_predicted_mask = 0

        for batch in generator_tqdm:
            batch_count += 1
            batch = util.move_to_device(batch, cuda_device)
            batch_results = model(**batch)
            loss = batch_results.get("loss")

            labels = batch['label']
            premise = batch['premise']['tokens']
            batch_size = premise.shape[0]

            label_probs = batch_results['label_probs']
            preds = label_probs.max(1)[1]
            neutral_probs = label_probs[:,args.target_label]

            deleted_inds = batch_results['deleted_inds']

            succeed = (preds == args.target_label)

            for i, sents in enumerate(batch['metadata']):
                gold_label = labels[i].item()
                neutral_prob = neutral_probs[i].item()
                #hyp = ' '.join(sents['hypothesis_tokens'][:-1])
                hyp = ' '.join(sents['hypothesis_tokens'][:])

                rem_inds = torch.nonzero(deleted_inds[i] > 0.5).cpu().numpy()
                rem_inds = rem_inds.squeeze().tolist()
                if type(rem_inds) == int:
                    rem_inds = [rem_inds]

                #prem_words = sents['premise_tokens'][:-1]
                prem_words = sents['premise_tokens'][:]
                prem = ' '.join(prem_words)
                prem_rm_words = []
                for ind, word in enumerate(prem_words):
                    if deleted_inds[i][ind] > 0.5:
                        prem_rm_words.append('$$')
                    else:
                        prem_rm_words.append(word)

                prem_rm = ' '.join(prem_rm_words)

                suc = succeed[i].item()

                out_dict = {'sentence1': prem,
                            'sentence2': hyp,
                            'sentence1_masked': prem_rm,
                            'masked_inds': rem_inds,
                            'num_masked': len(rem_inds),
                            'gold_label': label_dict[gold_label],
                            'label_prob': neutral_prob,
                            'succeed': suc,
                            'id': sents['id'],
                            'evidence': sents['evidence']
                            }

                if 'premise_delete' in sents:
                    teacher_mask = sents['premise_delete']
                    if teacher_mask is not None:
                        out_dict['teacher_mask'] = teacher_mask

                        true_num = len(teacher_mask)
                        true_covered = len(set(teacher_mask) & set(rem_inds))

                        predicted_num = len(rem_inds)

                        tot_teacher_mask += true_num
                        tot_teacher_covered += true_covered
                        tot_predicted_mask += predicted_num

                if DBG_OUTPUT_BY_DIST:
                    gold_del = batch['premise_delete'][i]
                    recall = torch.mul(gold_del, deleted_inds[i])
                    dist = torch.dist(recall, gold_del, p=1).item()
                    if dist < 1:
                        continue
                    else:
                        out_dict['dist_from_gold'] = dist

                out_writer.write(out_dict)

            corrects = torch.sum(succeed).item()

            total_corrects += corrects
            total_count += batch_size

            metrics = model.get_metrics()

            if loss is not None:
                loss_count += 1
                metrics["loss"] = loss.item()
                total_loss += loss.item()

            if (not _warned_tqdm_ignores_underscores and
                        any(metric_name.startswith("_") for metric_name in metrics)):
                print("Metrics with names beginning with \"_\" will "
                               "not be logged to the tqdm progress bar.")
                _warned_tqdm_ignores_underscores = True
            description = ', '.join(["%s: %.2f" % (name, value) for name, value
                                     in metrics.items() if not name.startswith("_")]) + " ||"
            generator_tqdm.set_description(description, refresh=False)

        final_metrics = model.get_metrics(reset=True)
        if loss_count > 0:
            if loss_count != batch_count:
                raise RuntimeError("The model you are trying to evaluate only sometimes " +
                                   "produced a loss!")
            final_metrics["loss"] = total_loss/batch_count

        print("Modified to neutral accuracy: {:.2f}".format(total_corrects/ total_count))

        if tot_teacher_mask > 0 :
            recall = tot_teacher_covered / tot_teacher_mask
            prec = tot_teacher_covered / tot_predicted_mask
            f1 = 2*recall*prec / (recall + prec)
            print("Recall: {:.2f} ( {} / {} ), Precision: {:.2f} ( {} / {} ), F1: {:.2f}".format(recall, tot_teacher_covered, tot_teacher_mask, prec, tot_teacher_covered, tot_predicted_mask, f1))
            final_metrics['recall'] = recall
            final_metrics['precision'] = prec
            final_metrics['F1'] = f1

        out_writer.close()
        return final_metrics

def evaluate_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    # Disable some of the more verbose logging statements
    #logging.getLogger('allennlp.common.params').disabled = True
    #logging.getLogger('allennlp.nn.initializers').disabled = True
    #logging.getLogger('allennlp.modules.token_embedders.embedding').setLevel(logging.INFO)

    # Load from archive
    archive = load_archive(args.archive_file, args.cuda_device, args.overrides, args.weights_file)
    config = archive.config
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data

    # Try to use the validation dataset reader if there is one - otherwise fall back
    # to the default dataset_reader used for both training and validation.
    validation_dataset_reader_params = config.pop('validation_dataset_reader', None)
    if validation_dataset_reader_params is not None:
        dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
    else:
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))
    evaluation_data_path = args.input_file
    print("Reading evaluation data from %s", evaluation_data_path)
    instances = dataset_reader.read(evaluation_data_path)

    iterator_params = config.pop("validation_iterator", None)
    if iterator_params is None:
        iterator_params = config.pop("iterator")
    iterator = DataIterator.from_params(iterator_params)
    iterator.index_with(model.vocab)

    metrics = evaluate(model, instances, iterator, args.cuda_device, args.preds_file)

    print("Finished evaluating.")
    print("Metrics:")
    for key, metric in metrics.items():
        print("%s: %s", key, metric)

    output_file = args.log_file
    if output_file:
        with open(output_file, "w") as file:
            json.dump(metrics, file, indent=4)
    return metrics


if __name__ == '__main__':
    if not args.append:
        assert(not os.path.exists(args.preds_file))

    dir_name, file_name = os.path.split(args.preds_file)
    os.makedirs(dir_name, exist_ok=True)

    file_path, file_ext = os.path.splitext(args.preds_file)
    log_file_ext = '.log'
    log_file = '{}{}'.format(file_path, log_file_ext)

    input_args = argparse.Namespace()
    input_args.cuda_device = args.cuda_device
    input_args.archive_file = args.archive_file
    input_args.input_file = args.input_file
    input_args.overrides = ''
    input_args.overrides = ''
    input_args.weights_file = args.weights_file
    input_args.output_file = ''
    input_args.preds_file = args.preds_file
    input_args.log_file = log_file
    input_args.target_label = args.target_label
    metrics = evaluate_from_args(input_args)
    print(metrics)
