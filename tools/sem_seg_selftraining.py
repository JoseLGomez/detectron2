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
import sys
import glob
import random
import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import time
import math
import PIL.Image as Image
import datetime
import itertools
import gc
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
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
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.generic_sem_seg_dataset import load_dataset_from_txt, load_dataset_to_inference, load_dataset_from_txt_and_merge
from torch import nn
import torch
from contextlib import ExitStack, contextmanager
from cityscapesscripts.helpers.labels import trainId2label, labels
from detectron2.utils.logger import log_every_n_seconds
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler

logger = logging.getLogger("detectron2")
INIT_TGT_PORT = 0.2
MAX_TGT_PORT = 0.5
TGT_PORT_STEP = 0.05
DS_RATE = 4
softmax2d = nn.Softmax2d()

def cotraining_argument_parser(parser):
    # Adds cotrainig arguments to the detectron2 base parser
    parser.add_argument(
        '--unlabeled_dataset_A',
        dest='unlabeled_dataset_A',
        help='File with Data A images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--unlabeled_dataset_A_name',
        dest='unlabeled_dataset_A_name',
        help='Unlabeled dataset name to call dataloader function',
        default=None,
        type=str
    )    
    parser.add_argument(
        '--weights_branchA',
        dest='weights_branchA',
        help='Weights File of branch1',
        default=None,
        type=str
    )
    parser.add_argument(
        '--num-epochs',
        dest='epochs',
        help='Number of selftraining rounds',
        default=20,
        type=int
    )
    parser.add_argument(
        '--max_unlabeled_samples',
        dest='max_unlabeled_samples',
        help='Number of maximum unlabeled samples',
        default=500,
        type=int
    ) 
    parser.add_argument(
        '--samples',
        dest='samples',
        help='Number of top images to be sampled after each iteration',
        default=40,
        type=int
    )
    parser.add_argument(
        '--step_inc',
        dest='step_inc',
        help='Fix a image step to avoid consecutive images in secuences',
        default=1,
        type=int
    )
    parser.add_argument(
        '--continue_epoch',
        dest='continue_epoch',
        help='Continue co-training at the begining of the specified epoch',
        default=0,
        type=int
    )
    parser.add_argument(
        '--continue_training',
        action='store_true'
    )
    parser.add_argument(
        '--incremental_training',
        help='Models are not reset in each epoch',
        action='store_true'
    )
    parser.add_argument(
        '--fp_annot',
        action='store_true'
    )
    parser.add_argument(
        '--initial_score_A',
        dest='initial_score_A',
        help='Initial score to reach to propagate weights to the next epoch',
        default=0,
        type=float
    )
    return parser

def print_txt_format(results_dict, iter_name, epoch, output, model_id):
    with open(os.path.join(output,'results.txt'),"a+") as f:
        print('----- Epoch: %s iteration: %s Model: %s -----' % (epoch,iter_name,model_id))
        f.write('----- Epoch: %s iteration: %s Model: %s ----- \n' % (epoch,iter_name,model_id))
        for k, v in results_dict['sem_seg'].items():
            if 'IoU' in k:
                print('%s: %.4f' % (k, v))
                f.write('%s: %.4f \n' % (k, v))
        print('\n')
        f.write('\n')

def built_custom_dataset(cfg, image_dir, gt_dir, dataset_name, add_pseudolabels=False, pseudo_img_dir=None, pseudo_dir=None, test=False):
    if add_pseudolabels and pseudo_img_dir is not None and pseudo_dir is not None:
        DatasetCatalog.register(
            dataset_name, lambda x1=image_dir, x2=pseudo_img_dir, y1=gt_dir, y2=pseudo_dir: load_dataset_from_txt_and_merge(x1, x2, y1, y2, num_samples=cfg.DATASETS.TRAIN_SAMPLES)
        )
    else:
        if test:
            DatasetCatalog.register(
                dataset_name, lambda x=image_dir, y=gt_dir: load_dataset_from_txt(x, y)
            )
        else:
            DatasetCatalog.register(
                dataset_name, lambda x=image_dir, y=gt_dir: load_dataset_from_txt(x, y, num_samples=cfg.DATASETS.TRAIN_SAMPLES)
            )
    if cfg.DATASETS.LABELS == 'cityscapes':
        MetadataCatalog.get(dataset_name).stuff_classes = [k.name for k in labels if k.trainId < 19 and k.trainId > -1]
        MetadataCatalog.get(dataset_name).stuff_colors = [k.color for k in labels if k.trainId < 19 and k.trainId > -1]
    else:
        raise Exception('Unsupported label set')
    MetadataCatalog.get(dataset_name).set(
        image_dir=image_dir,
        gt_dir=gt_dir,
        evaluator_type="generic_sem_seg",
        ignore_label=255,
    )

def built_inference_dataset(cfg, im_list, dataset_name):
    DatasetCatalog.register(
        dataset_name, lambda x=im_list: load_dataset_to_inference(x)
    )
    MetadataCatalog.get(dataset_name).set(
        image_dir=im_list,
        evaluator_type="generic_sem_seg",
        ignore_label=255,
    )


def build_sem_seg_train_aug(cfg, augmentation):
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
    if augmentation['hflip']:
        augs.append(T.RandomFlip(prob=augmentation['hflip_prob'], horizontal=True, vertical=False))
    if augmentation['vflip']:
        augs.append(T.RandomFlip(prob=augmentation['vflip_prob'], horizontal=False, vertical=True))
    if augmentation['cutout']:
        augs.append(T.CutOutPolicy(augmentation['cutout_n_holes'], augmentation['cutout_length']))
    if augmentation['random_resize']:
        augs.append(T.TrainScalePolicy(augmentation['resize_range']))
    return augs


def get_evaluator(cfg, dataset_name, output_folder=None):
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
        return SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder, write_outputs=False)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test_txt(cfg, model, dataset_name, step_iter, epoch, model_id):
    results = OrderedDict()
    dataset: List[Dict] = DatasetCatalog.get(dataset_name)
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(
        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, str(step_iter))
    )
    results_i = inference_on_dataset(model, data_loader, evaluator)
    results[dataset_name] = results_i
    print_txt_format(results_i, step_iter, epoch, cfg.OUTPUT_DIR, model_id)
    '''if comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_csv_format(results_i)'''
    if len(results) == 1:
        results = list(results.values())[0]
    return results

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


def inference_on_imlist(cfg, model, weights, dataset_name):
    # Following the same detectron2.evaluation.inference_on_dataset function
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).load(weights)  
    data_loader = build_detection_test_loader(cfg, dataset_name)
    total = len(data_loader)
    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())
        outputs = []
        pred_cls_num = np.zeros(num_classes)
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            batch_outputs = model(inputs)
            for output in batch_outputs:
                # Saving indexes and values of maximums instead the 20 channels scores to save memory
                output = output['sem_seg']
                output = torch.unsqueeze(output, 0)
                output = softmax2d(output)
                output = torch.squeeze(output)
                output = output.cpu().numpy()
                amax_output = np.asarray(np.argmax(output, axis=0), dtype=np.uint8)
                conf = np.amax(output,axis=0)
                outputs.append([amax_output, conf])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
    return outputs


def do_train(cfg, model, weights, train_dataset_name, test_dataset_name, model_id, save_checkpoints_path, epoch, 
             continue_epoch, incremental_training=False, resume=False, dataset_pseudolabels=None):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, save_checkpoints_path, optimizer=optimizer, scheduler=scheduler
    )
    max_iter = cfg.SOLVER.MAX_ITER
    if resume:
        start_iter = (
                checkpointer.resume_or_load(weights, resume=resume).get("iteration", -1) + 1
            )
    else:
        if not incremental_training or continue_epoch == 0:
            checkpointer.resume_or_load(weights)
        elif incremental_training:
            previous_save_checkpoints_path = save_checkpoints_path.split('/')
            previous_save_checkpoints_path[-2] = str(continue_epoch - 1)
            weights = os.path.join('/'.join(previous_save_checkpoints_path),'model_final.pth')
            checkpointer.resume_or_load(weights)

        start_iter = 0

    periodic_checkpointer = PeriodicCheckpointer(
            checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
        )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    #Data aug mapper
    augmentations = {}
    if model_id.lower() == 'a':
        augmentations = {'cutout': cfg.AUGMENTATION_A.CUTOUT,
                         'cutout_n_holes': cfg.AUGMENTATION_A.CUTOUT_N_HOLES,
                         'cutout_length': cfg.AUGMENTATION_A.CUTOUT_LENGTH,
                         'hflip': cfg.AUGMENTATION_A.HFLIP,
                         'hflip_prob': cfg.AUGMENTATION_A.HFLIP_PROB,
                         'vflip': cfg.AUGMENTATION_A.VFLIP,
                         'vflip_prob': cfg.AUGMENTATION_A.VFLIP_PROB,
                         'random_resize': cfg.AUGMENTATION_A.RANDOM_RESIZE,
                         'resize_range': cfg.AUGMENTATION_A.RESIZE_RANGE}
    elif model_id.lower() == 'b':
        augmentations = {'cutout': cfg.AUGMENTATION_B.CUTOUT,
                         'cutout_n_holes': cfg.AUGMENTATION_B.CUTOUT_N_HOLES,
                         'cutout_length': cfg.AUGMENTATION_B.CUTOUT_LENGTH,
                         'hflip': cfg.AUGMENTATION_B.HFLIP,
                         'hflip_prob': cfg.AUGMENTATION_B.HFLIP_PROB,
                         'vflip': cfg.AUGMENTATION_B.VFLIP,
                         'vflip_prob': cfg.AUGMENTATION_B.VFLIP_PROB,
                         'random_resize': cfg.AUGMENTATION_B.RANDOM_RESIZE,
                         'resize_range': cfg.AUGMENTATION_B.RESIZE_RANGE}
    else:
        raise NotImplementedError('Unknown model id for data augmentation')

    
    if "SemanticSegmentor" in cfg.MODEL.META_ARCHITECTURE:
        mapper = DatasetMapper(cfg, is_train=True, augmentations=build_sem_seg_train_aug(cfg, augmentations))
    else:
        mapper = None

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        results_list = []
        if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS and dataset_pseudolabels is not None:
            dataset: List[Dict] = DatasetCatalog.get(train_dataset_name)
            data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[0])
            dataset_pseudo: List[Dict] = DatasetCatalog.get(dataset_pseudolabels)
            data_loader_pseudo = build_detection_train_loader(cfg, dataset=dataset_pseudo, mapper=mapper, total_batch_size=cfg.SOLVER.SOURCE_PSEUDOLABELS_BATCH_RATIO[1])
            results_list = training_loop_mixdatasets(cfg, model, start_iter, max_iter, data_loader, data_loader_pseudo, storage, optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name, epoch, model_id)
        else:
            dataset: List[Dict] = DatasetCatalog.get(train_dataset_name)
            data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper, total_batch_size=cfg.SOLVER.IMS_PER_BATCH)
            results_list = training_loop(cfg, model, start_iter, max_iter, data_loader, storage, optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name, epoch, model_id)
    return results_list

def training_loop(cfg, model, start_iter, max_iter, data_loader, storage, optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name, epoch, model_id):
    results_list = []
    for data, iteration in zip(data_loader, range(start_iter, max_iter)):
        #print(data[0]['file_name'])
        #print('%s x %s' % (data[0]['height'], data[0]['width']))
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
            results = do_test_txt(cfg, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()

        if iteration - start_iter > 5 and (
            (iteration + 1) % 20 == 0 or iteration == max_iter - 1
        ):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)
    return results_list

def training_loop_mixdatasets(cfg, model, start_iter, max_iter, data_loader, data_loader_pseudo, storage, optimizer, scheduler, periodic_checkpointer, writers, test_dataset_name, epoch, model_id):
    ''' Training loop that mixes two dataloaders to compose the final batch with the proportion specified'''
    results_list = []

    for data1, data2, iteration in zip(data_loader, data_loader_pseudo, range(start_iter, max_iter)):
        #print(data[0]['file_name'])
        #print('%s x %s' % (data[0]['height'], data[0]['width']))
        storage.iter = iteration
        data = data1+data2
        loss_dict = model(data)
        del data
        gc.collect()
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
            results = do_test_txt(cfg, model, test_dataset_name, iteration+1, epoch, model_id)
            results_list.append([results['sem_seg']['mIoU'],iteration])
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()

        if iteration - start_iter > 5 and (
            (iteration + 1) % 20 == 0 or iteration == max_iter - 1
        ):
            for writer in writers:
                writer.write()
        periodic_checkpointer.step(iteration)
    return results_list

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
    ) 
    return cfg

def get_unlabeled_data(unlabeled_dataset, step_inc, seed, samples):
    with open(unlabeled_dataset,'r') as f:
        im_list = [line.rstrip().split(' ') for line in f.readlines()]
    im_list.sort()
    init_indx = random.randrange(0, step_inc)
    indx_sampled = np.asarray(range(init_indx, len(im_list), step_inc), dtype=int)
    im_list = np.asarray(im_list)[indx_sampled]
    random.seed(seed)
    im_list = random.sample(im_list.tolist(), min(len(im_list), samples))
    return im_list

def prepare_confidence_maps(outputs, num_classes):
    conf_dict = {k: [] for k in range(num_classes)}
    pred_cls_num = np.zeros(num_classes)
    for idx, output in enumerate(outputs):
        amax_output = output[0]
        conf = output[1]
        pred_label = amax_output.copy()
        for idx_cls in range(num_classes):
            idx_temp = pred_label == idx_cls
            pred_cls_num[idx_cls] = pred_cls_num[idx_cls] + np.sum(idx_temp)
            if idx_temp.any():
                conf_cls_temp = conf[idx_temp].astype(np.float32)
                len_cls_temp = conf_cls_temp.size
                # downsampling by ds_rate
                conf_cls = conf_cls_temp[0:len_cls_temp:DS_RATE]
                conf_dict[idx_cls].extend(conf_cls)
    return conf_dict, pred_cls_num

def compute_kc_thresholds(conf_dict, pred_cls_num, tgt_portion, num_classes):
    # threshold for each class
    cls_thresh = np.ones(num_classes,dtype = np.float32)
    cls_sel_size = np.zeros(num_classes, dtype=np.float32)
    cls_size = np.zeros(num_classes, dtype=np.float32)
    for idx_cls in np.arange(0, num_classes):
        cls_size[idx_cls] = pred_cls_num[idx_cls]
        if conf_dict[idx_cls] != None:
            conf_dict[idx_cls].sort(reverse=True) # sort in descending order
            len_cls = len(conf_dict[idx_cls])
            cls_sel_size[idx_cls] = int(math.floor(len_cls * tgt_portion))
            len_cls_thresh = int(cls_sel_size[idx_cls])
            if len_cls_thresh != 0:
                cls_thresh[idx_cls] = conf_dict[idx_cls][len_cls_thresh-1]
            conf_dict[idx_cls] = None
    logger.info("CBST thresholds: {}".format(cls_thresh))
    return cls_thresh


def generate_pseudolabels(outputs, cls_thresh, tgt_num):
    pseudolabels = []
    pseudolabels_not_filtered = []
    scores_list = []
    for idx in range(tgt_num):
        pseudolabels_not_filtered.append(outputs[idx][0])
        pseudolabel = outputs[idx][0].copy()
        pred_conf = outputs[idx][1]
        weighted_conf = np.zeros(pred_conf.shape,dtype=np.float32)
        for idx in range(len(cls_thresh)):
            weighted_conf = weighted_conf + (pred_conf*(pseudolabel == idx)/cls_thresh[idx])
        pseudolabel[weighted_conf < 1] = 19 # '255' in cityscapes indicates 'unlabaled' for trainIDs
        pseudolabels.append(pseudolabel)
        #Compute image score using the mean of the weighted confidence pixels values higher than the threshold cls_thresh
        weighted_conf[weighted_conf < 1] = np.nan
        score = np.nanmean(weighted_conf)
        scores_list.append(score)
    return pseudolabels, scores_list, pseudolabels_not_filtered


def apply_cbst(outputs, num_classes, tgt_num, tgt_portion):
    conf_dict, pred_cls_num = prepare_confidence_maps(outputs, num_classes)
    cls_thresh = compute_kc_thresholds(conf_dict, pred_cls_num, tgt_portion, num_classes)
    pseudolabels, scores_list, pseudolabels_not_filtered = generate_pseudolabels(outputs, cls_thresh, tgt_num)
    return pseudolabels, scores_list, pseudolabels_not_filtered


def compute_mtp_thresholds(pred_conf, pred_cls_num, tgt_portion, num_classes):
    thres = []
    for i in range(num_classes):
        x = pred_conf[pred_cls_num==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*tgt_portion))])
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    return thres


def apply_mtp(outputs, num_classes, tgt_num, tgt_portion):
    pred_cls_num = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.uint8)
    pred_conf = np.zeros((tgt_num, outputs[0][0].shape[0], outputs[0][0].shape[1]), dtype=np.float32)
    for index, output in enumerate(outputs):
        pred_cls_num[index] = output[0]
        pred_conf[index] = output[1]
    thres = compute_mtp_thresholds(pred_conf, pred_cls_num, tgt_portion, num_classes)
    logger.info("MPT thresholds: {}".format(thres))
    pseudolabels = []
    pseudolabels_not_filtered = []
    scores_list = []
    for index in range(tgt_num):
        pseudolabels_not_filtered.append(pred_cls_num[index])
        label = pred_cls_num[index].copy()
        prob = pred_conf[index]
        for i in range(num_classes):
            label[(prob<thres[i])*(label==i)] = 19 # '255' in cityscapes indicates 'unlabaled' for trainIDs
            prob[(prob<thres[i])*(label==i)] = np.nan
        pseudolabels.append(label)
        score = np.nanmean(prob)
        scores_list.append(score)
    return pseudolabels, scores_list, pseudolabels_not_filtered


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_pseudolabels(images, pseudolabels, scores, pseudolabels_path, coloured_pseudolabels_path, 
                      pseudolabels_not_filtered=None, coloured_pseudolabels_not_filtered_path=None):
    filenames_and_scores = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'filenames_and_scores.txt')
    images_txt = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'selected_images_path.txt')
    psedolabels_txt = os.path.join('/'.join(pseudolabels_path.split('/')[:-1]),'selected_pseudolabels_path.txt')
    with open(filenames_and_scores,'w') as f:
        with open(images_txt,'w') as g:
            with open(psedolabels_txt,'w') as h:
                for idx, image in enumerate(images):
                    filename = image[0].split('/')[-1]
                    Image.fromarray(pseudolabels[idx]).save(os.path.join(pseudolabels_path,filename))
                    pred_colour = 255 * np.ones([pseudolabels[idx].shape[0],pseudolabels[idx].shape[1],3], dtype=np.uint8)
                    for train_id, label in trainId2label.items():
                        pred_colour[(pseudolabels[idx] == train_id),0] = label.color[0]
                        pred_colour[(pseudolabels[idx] == train_id),1] = label.color[1]
                        pred_colour[(pseudolabels[idx] == train_id),2] = label.color[2]
                    Image.fromarray(pred_colour).save(os.path.join(coloured_pseudolabels_path,filename))
                    if pseudolabels_not_filtered is not None and coloured_pseudolabels_not_filtered_path is not None:
                        pred_colour = 255 * np.ones([pseudolabels_not_filtered[idx].shape[0],pseudolabels_not_filtered[idx].shape[1],3], dtype=np.uint8)
                        for train_id, label in trainId2label.items():
                            pred_colour[(pseudolabels_not_filtered[idx] == train_id),0] = label.color[0]
                            pred_colour[(pseudolabels_not_filtered[idx] == train_id),1] = label.color[1]
                            pred_colour[(pseudolabels_not_filtered[idx] == train_id),2] = label.color[2]
                        Image.fromarray(pred_colour).save(os.path.join(coloured_pseudolabels_not_filtered_path,filename))
                    #Create txt with files names and scores
                    f.write('%s %s\n' % (filename, str(scores[idx])))
                    g.write('%s\n' % (image[0]))
                    h.write('%s\n' % (os.path.join(pseudolabels_path,filename)))
    return images_txt, psedolabels_txt, filenames_and_scores

def merge_txts_and_save(new_txt, txt1, txt2=None):
    if txt2 is not None:
        files = [txt1, txt2]
    else:
        files = [txt1]
    with open(new_txt, 'w') as f:
        for file in files:
            with open(file) as infile:
                for line in infile:
                    f.write(line)
    return new_txt

def update_best_score_txts_and_save(accum_scores_txt, accum_images_txt, accum_labels_txt, new_scores_txt, 
                                    new_images_txt, new_labels_txt, save_img_txt, save_labels_txt, save_scores_txt):
    with open(accum_scores_txt,'r') as f:
        accum_scores = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_scores_txt,'r') as f:
        new_scores_txt = [line.rstrip().split(' ') for line in f.readlines()]
    with open(accum_images_txt,'r') as f:
        accum_images = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_images_txt,'r') as f:
        new_images = [line.rstrip().split(' ') for line in f.readlines()]
    with open(accum_labels_txt,'r') as f:
        accum_labels = [line.rstrip().split(' ') for line in f.readlines()]
    with open(new_labels_txt,'r') as f:
        new_labels = [line.rstrip().split(' ') for line in f.readlines()]
    ignore_list = []
    # Check for repeated images
    for idx, score in enumerate(new_scores_txt):
        for idx2, score2 in enumerate(accum_scores):
            if score[0] == score2[0]: 
                if score[1] > score2[1]:
                    # If we found the same image with better score we updated values in all the acumulated lists
                    accum_scores[idx2][1] = score[1]
                    accum_labels[idx2] = new_labels[idx]
                # we store the index to do not add it again later
                ignore_list.append(idx)
                break
    # add new images into the accumulated ones
    for idx, score in enumerate(new_scores_txt):
        if idx not in ignore_list:
            accum_scores.append(score)
            accum_labels.append(new_labels[idx])
            accum_images.append(new_images[idx])
    # save each data in its respective txt
    new_img_dataset = open(save_img_txt,'w')
    new_labels_dataset = open(save_labels_txt,'w')
    new_scores_dataset = open(save_scores_txt,'w')
    for idx, _ in enumerate(accum_scores):
        new_img_dataset.write(accum_images[idx][0] + '\n')
        new_labels_dataset.write(accum_labels[idx][0] + '\n')
        new_scores_dataset.write(accum_scores[idx][0] + ' ' + accum_scores[idx][1] + '\n')
    new_img_dataset.close()
    new_labels_dataset.close()
    new_scores_dataset.close()
    return save_img_txt, save_labels_txt, save_scores_txt
    

def main(args):
    cfg = setup(args)
    continue_epoch = args.continue_epoch 
    accumulated_selection_imgA = []
    accumulated_selection_pseudoA = []
    accumulated_acores_A = []
    pseudolabeling = cfg.PSEUDOLABELING.MODE
    collaboration = cfg.PSEUDOLABELING.COLLABORATION
    accumulation_mode = cfg.PSEUDOLABELING.ACCUMULATION
    num_selected = cfg.PSEUDOLABELING.NUMBER
    weights_branchA = args.weights_branchA
    if pseudolabeling == 'cbst':
        tgt_portion = INIT_TGT_PORT
    # Set initial scores to surpass during an epoch to propagate weghts to the next one
    best_score_A = args.initial_score_A

    # Build test dataset
    built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, cfg.DATASETS.TEST_NAME, test=True)

    # Start self-training
    for epoch in range(args.continue_epoch,args.epochs):
        logger.info("Starting training from iteration {}".format(epoch))
        # prepare unlabeled data
        logger.info("prepare unlabeled data")
        seed = random.randrange(sys.maxsize)
        # Read unlabeled data from the txt specified and select randomly X samples defined on max_unlabeled_samples
        unlabeled_datasetA = get_unlabeled_data(args.unlabeled_dataset_A, args.step_inc, seed, args.max_unlabeled_samples)
        logger.info("Unlabeled data selected from {}: {}".format(args.unlabeled_dataset_A,len(unlabeled_datasetA)))
        # Regiter unlabeled dataset on detectron 2
        built_inference_dataset(cfg, unlabeled_datasetA, args.unlabeled_dataset_A_name)
        # Compute inference on unlabeled datasets
        model = build_model(cfg)
        logger.info("Compute inference on unlabeled datasets")
        start_time = time.perf_counter()
        # Inference return a tuple of labels and confidences
        inference_A = inference_on_imlist(cfg, model, weights_branchA, args.unlabeled_dataset_A_name)
        total_time = time.perf_counter() - start_time
        logger.info("Compute inference on unlabeled dataset A: {:.2f} s".format(total_time))
        logger.info("Pseudolabeling mode: {}".format(pseudolabeling)) 
        if pseudolabeling == 'cbst':
            start_time = time.perf_counter()
            pseudolabels_A, scores_listA, pseudolabels_A_not_filtered = apply_cbst(inference_A, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,len(unlabeled_datasetA),tgt_portion)
            total_time = time.perf_counter() - start_time
            logger.info("CBST on unlabeled dataset A: {:.2f} s".format(total_time))
            tgt_portion = min(tgt_portion + TGT_PORT_STEP, MAX_TGT_PORT)
        elif pseudolabeling == 'mpt':
            start_time = time.perf_counter()
            pseudolabels_A, scores_listA, pseudolabels_A_not_filtered = apply_mtp(inference_A, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,len(unlabeled_datasetA), 0.5)
            total_time = time.perf_counter() - start_time
            logger.info("MPT on unlabeled dataset A: {:.2f} s".format(total_time))
        else:
            raise Exception('unknown pseudolabeling method defined')

        #save pseudolabels
        pseudolabels_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling/pseudolabels')
        create_folder(pseudolabels_path_model_A)
        coloured_pseudolabels_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling/coloured_pseudolabels')
        create_folder(coloured_pseudolabels_path_model_A)
        coloured_pseudolabels_not_filtered_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling/coloured_pseudolabels_not_filtered')
        create_folder(coloured_pseudolabels_not_filtered_path_model_A)
        dataset_A_path = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'unlabeled_data_selected')
        create_folder(dataset_A_path)
        checkpoints_A_path = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'checkpoints')
        create_folder(checkpoints_A_path)

        # Continue cotraining on the specified epoch
        if continue_epoch > 0:
            accumulated_selection_imgA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),'unlabeled_data_selected/dataset_img.txt')
            accumulated_selection_pseudoA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),'unlabeled_data_selected/dataset_pseudolabels.txt')
            accumulated_scores_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch-1),'unlabeled_data_selected/filenames_and_scores.txt')
            continue_epoch = 0

        scores_listA = np.asarray(scores_listA)
        pseudolabels_A = np.asarray(pseudolabels_A)
        unlabeled_datasetA = np.asarray(unlabeled_datasetA)
        pseudolabels_A_not_filtered = np.asarray(pseudolabels_A_not_filtered)
        # Order pseudolabels by confidences (scores) higher to lower and select number defined to merge with source data
        sorted_idx = np.argsort(scores_listA)[::-1][:len(scores_listA)]
        sorted_scores_listA = scores_listA[sorted_idx]
        sorted_pseudolabels_A = pseudolabels_A[sorted_idx]
        sorted_unlabeled_datasetA = unlabeled_datasetA[sorted_idx]
        sorted_pseudolabels_A_not_filtered = pseudolabels_A_not_filtered[sorted_idx]

        # free memory
        del scores_listA
        del pseudolabels_A
        del unlabeled_datasetA
        del pseudolabels_A_not_filtered
        gc.collect()

        logger.info("Select candidates and Save on disk")
        # select candidates and save them to add them to the source data
        if len(sorted_unlabeled_datasetA) > cfg.PSEUDOLABELING.NUMBER:
            images_txt_A, psedolabels_txt_A, filenames_and_scoresA = save_pseudolabels(sorted_unlabeled_datasetA[:num_selected], sorted_pseudolabels_A[:num_selected], sorted_scores_listA[:num_selected], 
                pseudolabels_path_model_A, coloured_pseudolabels_path_model_A, sorted_pseudolabels_A_not_filtered[:num_selected], coloured_pseudolabels_not_filtered_path_model_A)     
        else:
            images_txt_A, psedolabels_txt_A, filenames_and_scoresA = save_pseudolabels(sorted_unlabeled_datasetA, sorted_pseudolabels_A, sorted_scores_listA, pseudolabels_path_model_A, 
                coloured_pseudolabels_path_model_A, sorted_pseudolabels_A_not_filtered, coloured_pseudolabels_not_filtered_path_model_A)

        # free memory
        del sorted_unlabeled_datasetA
        del sorted_pseudolabels_A
        del sorted_scores_listA
        del sorted_pseudolabels_A_not_filtered
        gc.collect()

        # Compute data accumulation procedure
        logger.info("Compute data accumulation procedure selected: {}".format(accumulation_mode))
        if accumulation_mode is not None and len(accumulated_selection_imgA) > 0:
            if accumulation_mode.lower() == 'all':
                accumulated_selection_imgA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_img.txt'),
                                                                    accumulated_selection_imgA, images_txt_A)
                accumulated_selection_pseudoA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                                    accumulated_selection_pseudoA, psedolabels_txt_A)
                accumulated_scores_A = merge_txts_and_save(os.path.join(dataset_A_path,'filenames_and_scores.txt'),
                                                                    accumulated_scores_A, filenames_and_scoresA)
            if accumulation_mode.lower() == 'update_best_score':
                accumulated_selection_imgA, accumulated_selection_pseudoA, accumulated_scores_A = update_best_score_txts_and_save(
                                                accumulated_scores_A, accumulated_selection_imgA, accumulated_selection_pseudoA, 
                                                filenames_and_scoresA, images_txt_A, psedolabels_txt_A, 
                                                os.path.join(dataset_A_path,'dataset_img.txt'), 
                                                os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                os.path.join(dataset_A_path,'filenames_and_scores.txt'))
        else:
            #No accumulation, only training with new pseudolabels
            accumulated_selection_imgA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_img.txt'),
                                                                    images_txt_A)
            accumulated_selection_pseudoA = merge_txts_and_save(os.path.join(dataset_A_path,'dataset_pseudolabels.txt'),
                                                                    psedolabels_txt_A)
            accumulated_scores_A = merge_txts_and_save(os.path.join(dataset_A_path,'filenames_and_scores.txt'),
                                                                    filenames_and_scoresA)


        if cfg.SOLVER.ALTERNATE_SOURCE_PSEUDOLABELS:
            # create one dataloader for the source data and another for target pseudolabels 
            dataset_A_source = cfg.DATASETS.TRAIN_NAME + '_A_source' + str(epoch)
            dataset_A_target = cfg.DATASETS.TRAIN_NAME + '_A_target' + str(epoch)
            built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT, cfg.DATASETS.TRAIN_GT_TXT, dataset_A_source)
            built_custom_dataset(cfg, accumulated_selection_imgA, accumulated_selection_pseudoA, dataset_A_target)
            # Train model A
            logger.info("Training Model A")
            results_A = do_train(cfg, model, args.weights_branchA, dataset_A_source, cfg.DATASETS.TEST_NAME,'a', checkpoints_A_path, epoch, args.continue_epoch, 
                                 incremental_training=args.incremental_training, resume=False, dataset_pseudolabels=dataset_A_target)

            DatasetCatalog.remove(dataset_A_source)
            MetadataCatalog.remove(dataset_A_source)
            DatasetCatalog.remove(dataset_A_target)
            MetadataCatalog.remove(dataset_A_target)
        else:
            # create dataloader adding psedolabels to source dataset
            dataset_A_name = cfg.DATASETS.TRAIN_NAME + '_A_' + str(epoch)
            built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT, cfg.DATASETS.TRAIN_GT_TXT, 
                                                   dataset_A_name, True, accumulated_selection_imgA, accumulated_selection_pseudoA)
            # Train model A
            logger.info("Training Model A")
            results_A = do_train(cfg, model, args.weights_branchA, dataset_A_name, cfg.DATASETS.TEST_NAME,'a', checkpoints_A_path, epoch, args.continue_epoch, 
                                 incremental_training=args.incremental_training, resume=False)

            # delete all datasets registered during epoch
            DatasetCatalog.remove(dataset_A_name)
            MetadataCatalog.remove(dataset_A_name)

        DatasetCatalog.remove(args.unlabeled_dataset_A_name)
        MetadataCatalog.remove(args.unlabeled_dataset_A_name)

        # refresh weight file pointers after iteration for initial inference if there is improvement

        if not args.incremental_training:
            for score, iteration in results_A:
                if score > best_score_A:
                    best_score_A = score
                    weights_branchA = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'checkpoints/model_%s.pth' % (str(iteration).zfill(7)))
            logger.info("Best model A until now: {}".format(weights_branchA))
            logger.info("Best mIoU: {}".format(best_score_A))


if __name__ == "__main__":
    default_parser = default_argument_parser()
    args = cotraining_argument_parser(default_parser).parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
