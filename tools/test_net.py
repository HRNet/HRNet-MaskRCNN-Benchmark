# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os, sys

import tqdm
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from multiprocessing import Process, Manager


def calc_iou(a, b):
    # TODO: to remove
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x1 >= x2 or y1 >= y2:
        return 0
    inter = (x2-x1+1) * (y2-y1+1)
    union = (a[2]-a[0]+1)*(a[3]-a[1]+1) + (b[2]-b[0]+1)*(b[3]-b[1]+1)
    return inter / (union - inter)


def voting(predictions, local_rank, flip=False, num_process=20, overlap=0.7):

    if local_rank != 0:
        sys.exit(0)
    print("Processing BBox voting ..... ")
    manager = Manager()
    results = manager.list()
    num_samples = len(predictions[0])
    voted_predictions = predictions[-1]
    num_per_process = int(num_samples/num_process)
    ids = []
    for i in range(num_process):
        ids.append(num_per_process*i)
    ids.append(num_samples)
    procs = []
    for i in range(num_process):
        procs.append(Process(target=process_voting, args=(predictions, ids[i], ids[i+1], results, overlap, flip)))

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    print("Voting finished.....")
    print("Accumulating results: %s ...." % len(results))
    for i in tqdm.tqdm(range(len(results))):
        img_id, voted_bboxes, voted_labels, voted_scores = results[i]
        voted_predictions[img_id].bbox = torch.cat(voted_bboxes).reshape(len(voted_bboxes), 4)
        voted_predictions[img_id].extra_fields['labels'] = torch.Tensor(voted_labels)
        voted_predictions[img_id].extra_fields['scores'] = torch.Tensor(voted_scores)
    return voted_predictions


def process_voting(predictions, start_ind, end_ind, results, overlap=0.7, flip=False):
    print(start_ind, end_ind)
    num_scales = len(predictions)
    flip_scales = int(num_scales/2) if flip else num_scales
    # enumerate each image
    for img_id in range(start_ind, end_ind):
        num_bboxes = []
        image_width, image_height = predictions[-1][img_id].size
        for i in range(num_scales):
            predictions[i][img_id] = predictions[i][img_id].resize((image_width, image_height))

        for i in range(flip_scales, num_scales):
            predictions[i][img_id] = predictions[i][img_id].transpose(0)

        for i in range(num_scales):
            num_bboxes.append(predictions[i][img_id].bbox.size(0))

        valid = torch.zeros(sum(num_bboxes))
        bboxes = torch.Tensor(sum(num_bboxes), 4)
        labels = torch.Tensor(sum(num_bboxes))
        scores = torch.Tensor(sum(num_bboxes))

        for i in range(num_scales):
            bboxes[sum(num_bboxes[:i]):sum(num_bboxes[:i+1])] = predictions[i][img_id].bbox
            labels[sum(num_bboxes[:i]):sum(num_bboxes[:i+1])] = predictions[i][img_id].extra_fields['labels']
            scores[sum(num_bboxes[:i]):sum(num_bboxes[:i+1])] = predictions[i][img_id].extra_fields['scores']

        # sort by scores
        _, indices = scores.sort(dim=0, descending=True)

        # voting, by enumerate each bbox
        voted_bboxes = []
        voted_labels = []
        voted_scores = []
        for idx in indices:
            if valid[idx] == 1:
                continue
            valid[idx] = 1
            tempbox_list = []

            bbox, label, score = bboxes[idx], labels[idx], scores[idx]

            for j in indices:
                if valid[j] == 1:
                    continue
                bboxj, labelj, scorej = bboxes[j], labels[j], scores[j]
                if labelj != label:
                    continue
                if calc_iou(bbox, bboxj) < overlap:
                    continue
                valid[j] = 1
                tempbox_list.append(bboxj)

            n = len(tempbox_list)
            for i in range(n):
                for j in range(4):
                    bbox[j] += tempbox_list[i][j]
            for j in range(4):
                bbox[j] /= n+1

            voted_bboxes.append(bbox)
            voted_labels.append(label)
            voted_scores.append(score)
        if sum(num_bboxes) != 0:
            results.append((img_id, voted_bboxes, voted_labels, voted_scores))

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder

    if cfg.TEST.MULTI_SCALE:
        data_loaders_val = []
        for min_size_test, max_size_test in cfg.TEST.MULTI_SIZES:
            cfg.defrost()
            cfg.INPUT.MIN_SIZE_TEST = min_size_test
            cfg.INPUT.MAX_SIZE_TEST = max_size_test
            cfg.freeze()
            data_loaders_val.extend(make_data_loader(cfg, is_train=False, is_distributed=distributed))
        output_folders = output_folders * len(cfg.TEST.MULTI_SIZES)
        dataset_names = dataset_names * len(cfg.TEST.MULTI_SIZES)
    else:
        data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    predictions = []

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        prediction = inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
        predictions.append(prediction)

    if cfg.TEST.MULTI_SCALE:

        logger.info("Processing multi-scale bbox voting....")
        voted_predictions = voting(predictions, args.local_rank)  # box_voting(predictions, args.local_rank)
        torch.save(voted_predictions, os.path.join(output_folders[0], 'predictions.pth'))

        extra_args = dict(
            box_only=cfg.MODEL.RPN_ONLY,
            iou_types=iou_types,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )

        evaluate(dataset=data_loaders_val[0].dataset,
                 predictions=voted_predictions,
                 output_folder=output_folders[0],
                 **extra_args)

    else:
        for prediction, output_folder, dataset_name, data_loader_val in zip(predictions, output_folders, dataset_names,
                                                                            data_loaders_val):
            extra_args = dict(
                box_only=cfg.MODEL.RPN_ONLY,
                iou_types=iou_types,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            )

            evaluate(dataset=data_loader_val.dataset,
                     predictions=prediction,
                     output_folder=output_folder,
                     **extra_args)
    return 0


if __name__ == "__main__":
    main()
