#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import PIL.Image as Image # to debug
import numpy as np

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    DatasetMapper,
)
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    SemSegEvaluator_opt2,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data.datasets.generic_sem_seg_dataset import load_dataset_from_txt
from detectron2.data.common import MapDataset
from cityscapesscripts.helpers.labels import labels
from detectron2.data.catalog import Metadata
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

logger = logging.getLogger("detectron2")
test_dataset_name = None

def print_txt_format(results_dict, iter_name, output):
    with open(os.path.join(output,'results.txt'),"a+") as f:
        print('----- iteration: %s -----' % iter_name)
        f.write('----- iteration: %s ----- \n' % iter_name)
        for k, v in results_dict['sem_seg'].items():
            if 'IoU' in k:
                print('%s: %.4f' % (k, v))
                f.write('%s: %.4f \n' % (k, v))
        print('\n')
        f.write('\n')


def built_custom_dataset(cfg, image_dir, gt_dir, split):
    dataset_name = cfg.DATASETS.TRAIN_NAME + split
    DatasetCatalog.register(
        dataset_name, lambda x=image_dir, y=gt_dir: load_dataset_from_txt(x, y)
    )
    if cfg.DATASETS.LABELS == 'cityscapes':
        MetadataCatalog.get(dataset_name).stuff_classes = [k.name for k in labels if k.trainId < 19 and k.trainId > -1]
        MetadataCatalog.get(dataset_name).stuff_colors = [k.color for k in labels if k.trainId < 19 and k.trainId > -1]
    else:
        raise Exception('Unsupported label set')
    MetadataCatalog.get(dataset_name).set(
        image_dir=image_dir,
        gt_dir=gt_dir,
        evaluator_type="generic_sem_seg_opt2",
        ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
    )
    return dataset_name

def build_sem_seg_train_aug(cfg):
    augs = []
    if cfg.INPUT.ACTIVATE_MIN_SIZE_TRAIN:
        augs.append(T.ResizeShortestEdge(
            cfg.INPUT.MIN_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING))
    if cfg.INPUT.RESIZED:
        augs.append(T.Resize(cfg.INPUT.RESIZE_SIZE))
    if cfg.INPUT.CROP.ENABLED:
        augs.append(T.RandomCrop_CategoryAreaConstraint(
                cfg.INPUT.CROP.TYPE,
                cfg.INPUT.CROP.SIZE,
                cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE))
    if cfg.AUGMENTATION.HFLIP:
        augs.append(T.RandomFlip(prob=cfg.AUGMENTATION.HFLIP_PROB, horizontal=True, vertical=False))
    if cfg.AUGMENTATION.VFLIP:
        augs.append(T.RandomFlip(prob=cfg.AUGMENTATION.VFLIP_PROB, horizontal=False, vertical=True))
    if cfg.AUGMENTATION.CUTOUT:
        augs.append(T.CutOutPolicy(cfg.AUGMENTATION.CUTOUT_N_HOLES, cfg.AUGMENTATION.CUTOUT_LENGTH))
    if cfg.AUGMENTATION.RANDOM_RESIZE:
        augs.append(T.TrainScalePolicy(cfg.AUGMENTATION.RESIZE_RANGE))
    return augs


def get_evaluator(cfg, dataset_name, data_loader, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name, output_folder=output_folder)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if evaluator_type == "generic_sem_seg":
        return SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder, 
                                write_outputs=True, ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE)
    if evaluator_type == "generic_sem_seg_opt2":
        return SemSegEvaluator_opt2(dataset_name, distributed=True, output_dir=output_folder, 
                                write_outputs=True, ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE, 
                                tgt_num=len(data_loader), logger=logger)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model, step_iter):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, data_loader, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, step_iter)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def do_test_txt(cfg, model, dataset_name, step_iter):
    results = OrderedDict()
    dataset: List[Dict] = DatasetCatalog.get(dataset_name)
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(
        cfg, dataset_name, data_loader, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, str(step_iter))
    )
    results_i = inference_on_dataset(model, data_loader, evaluator)
    results[dataset_name] = results_i
    print_txt_format(results_i, step_iter, cfg.OUTPUT_DIR)
    '''if comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_csv_format(results_i)'''
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    global test_dataset_name
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    #Data aug mapper
    if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg))
    else:
        mapper = None

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    train_dataset_name = built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT, cfg.DATASETS.TRAIN_GT_TXT, '_train')
    test_dataset_name = built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, '_test')
    dataset: List[Dict] = DatasetCatalog.get(train_dataset_name)
    data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, total_batch_size=cfg.SOLVER.IMS_PER_BATCH)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            # debug data-aug
            '''print(data[0]['image'].cpu().numpy().transpose(1,2,0).shape)
            if not os.path.exists(os.path.join(cfg.OUTPUT_DIR,'debug')):
                os.makedirs(os.path.join(cfg.OUTPUT_DIR,'debug'))
            Image.fromarray(data[0]['image'].cpu().numpy().transpose(1,2,0).astype(np.uint8)).save(os.path.join(cfg.OUTPUT_DIR,'debug','image_%s.png' % iteration))
            Image.fromarray(data[0]['sem_seg'].cpu().numpy().astype(np.uint8)).save(os.path.join(cfg.OUTPUT_DIR,'debug','mask_%s.png' % iteration))
            exit(-1)'''
            if cfg.AUGMENTATION.CUTOUT:
                for idx, _ in enumerate(data):
                    data[idx]['mask'] = data[idx]['sem_seg'] != 200 # recover mask from the CutOut
                    data[idx]['sem_seg'][data[idx]['sem_seg'] == 200] = 19 # assign void class to the gt
            storage.iter = iteration
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()


            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
            ):
                do_test_txt(cfg, model, test_dataset_name, iteration+1)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg

def main(args):
    global test_dataset_name
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if not args.eval_only:
        distributed = comm.get_world_size() > 1
        if distributed:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        do_train(cfg, model, resume=args.resume)
    else:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        test_dataset_name = built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, '_test')
        iteration = cfg.MODEL.WEIGHTS.split('/')[-1].split('.')[0].split('_')[-1]
        do_test_txt(cfg, model, test_dataset_name, iteration)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
