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
    KITTIEvaluator,
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
        '--unlabeled_dataset_B',
        dest='unlabeled_dataset_B',
        help='File with Data B images',
        default=None,
        type=str
    )
    parser.add_argument(
        '--unlabeled_dataset_B_name',
        dest='unlabeled_dataset_B_name',
        help='Unlabeled dataset name to call dataloader function',
        default=None,
        type=str
    ) 
    parser.add_argument(
        '--same_domain',
        help='Set when the unlabeled domain for Data A and Data B is the same (i.e. rgb and mirrored rgb)',
        action='store_true'
    )    
    parser.add_argument(
        '--weights_branchA',
        dest='weights_branchA',
        help='Weights File of branch1',
        default=None,
        type=str
    )
    parser.add_argument(
        '--weights_branchB',
        dest='weights_branchB',
        help='Weights File of branch2',
        default=None,
        type=str
    )
    parser.add_argument(
        '--num-epochs',
        dest='epochs',
        help='Number of cotraining iterations',
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
        '--load_epoch',
        dest='load_epoch',
        help='Loading epoch',
        default=0,
        type=int
    )
    parser.add_argument(
        '--continue_training',
        action='store_true'
    )
    parser.add_argument(
        '--fp_annot',
        action='store_true'
    )
    return parser


def built_custom_dataset(cfg, image_dir, gt_dir, dataset_name, add_pseudolabels=False, pseudo_img_dir=None, pseudo_dir=None):
    if add_pseudolabels and pseudo_img_dir is not None and pseudo_dir is not None:
        DatasetCatalog.register(
            dataset_name, lambda x1=image_dir, x2=pseudo_img_dir, y1=gt_dir, y2=pseudo_dir: load_dataset_from_txt_and_merge(x1, x2, y1, y2)
        )
    else:
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


def build_sem_seg_train_aug(cfg):
    augs = []
    if cfg.INPUT.FLIP_PROB is not None:
        if cfg.INPUT.RANDOM_FLIP == "horizontal":
            augs.append(T.RandomFlip(prob=cfg.INPUT.FLIP_PROB, horizontal=True, vertical=False))
        elif cfg.INPUT.RANDOM_FLIP == "vertical":
            augs.append(T.RandomFlip(prob=cfg.INPUT.FLIP_PROB, horizontal=False, vertical=True))
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
    if evaluator_type == "kitti":
        return KITTIEvaluator(dataset_name, cfg, output_folder)
    if evaluator_type == "generic_sem_seg":
        return SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test_txt(cfg, model, dataset_name, step_iter):
    results = OrderedDict()
    dataset: List[Dict] = DatasetCatalog.get(dataset_name)
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(
        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name, str(step_iter))
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
                output = output['sem_seg'].cpu().numpy()
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


def do_train(cfg, model, weights, train_dataset_name, test_dataset_name, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )

    start_iter = (
        checkpointer.resume_or_load(weights, resume=resume).get("iteration", -1) + 1
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
    dataset: List[Dict] = DatasetCatalog.get(train_dataset_name)
    data_loader = build_detection_train_loader(cfg, dataset=dataset, mapper=mapper)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            #print(data[0]['file_name'])
            #print('%s x %s' % (data[0]['height'], data[0]['width']))
            #print(data[0]['sem_seg'].shape)
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
                and iteration != max_iter - 1
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
        #output = torch.unsqueeze(output, 0)
        #output = softmax2d(output).cpu().data[0].numpy()
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
    conf_tot = 0.0
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
    return cls_thresh

def generate_pseudolabels(outputs, cls_thresh, tgt_num):
    pseudolabels = []
    scores_list = []
    for idx in range(tgt_num):
        pseudolabel = outputs[idx][0]
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
    return pseudolabels, scores_list


def apply_cbst(outputs, num_classes, tgt_num, tgt_portion):
    conf_dict, pred_cls_num = prepare_confidence_maps(outputs, num_classes)
    cls_thresh = compute_kc_thresholds(conf_dict, pred_cls_num, tgt_portion, num_classes)
    pseudolabels, scores_list = generate_pseudolabels(outputs, cls_thresh, tgt_num)
    return pseudolabels, scores_list


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_pseudolabels(images, pseudolabels, scores, pseudolabels_path, coloured_pseudolabels_path):
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
                    #Create txt with files names and scores
                    f.write('%s %s\n' % (filename, str(scores[idx])))
                    g.write('%s\n' % (image[0]))
                    h.write('%s\n' % (os.path.join(pseudolabels_path,filename)))
    return images_txt, psedolabels_txt

def merge_txts_and_save(txt1, txt2, new_txt):
    files = [txt1, txt2]
    with open(new_txt, 'w') as f:
        for file in files:
            with open(file) as infile:
                for line in infile:
                    f.write(line)


def main(args):
    cfg = setup(args)
    accumulated_selection_imgA = []
    accumulated_selection_pseudoA = []
    accumulated_selection_imgB = []
    accumulated_selection_pseudoB = []
    pseudolabeling = cfg.PSEUDOLABELING.MODE
    collaboration = cfg.PSEUDOLABELING.COLLABORATION
    accumulation_mode = cfg.PSEUDOLABELING.ACCUMULATION
    num_selected = cfg.PSEUDOLABELING.NUMBER
    if pseudolabeling == 'cbst':
        tgt_portion = INIT_TGT_PORT

    # Build test dataset
    built_custom_dataset(cfg, cfg.DATASETS.TEST_IMG_TXT, cfg.DATASETS.TEST_GT_TXT, cfg.DATASETS.TEST_NAME)

    # Start co-training
    for epoch in range(args.load_epoch,args.epochs):
        logger.info("Starting training from iteration {}".format(epoch))
        # prepare unlabeled data
        logger.info("prepare unlabeled data")
        seed = random.randrange(sys.maxsize)
        unlabeled_datasetA = get_unlabeled_data(args.unlabeled_dataset_A, args.step_inc, seed, args.max_unlabeled_samples)
        if args.same_domain:
            unlabeled_datasetB = unlabeled_datasetA
        else:    
            unlabeled_datasetB = get_unlabeled_data(args.unlabeled_dataset_B, args.step_inc, seed, args.max_unlabeled_samples)
        
        logger.info("Unlabeled data selected from A: {}".format(len(unlabeled_datasetA)))
        logger.info("Unlabeled data selected from B: {}".format(len(unlabeled_datasetB)))
        # Regiter unlabeled dataset
        built_inference_dataset(cfg, unlabeled_datasetA, args.unlabeled_dataset_A_name)
        built_inference_dataset(cfg, unlabeled_datasetB, args.unlabeled_dataset_B_name)

        # Compute inference on unlabeled datasets
        model = build_model(cfg)
        logger.info("Compute inference on unlabeled datasets")
        start_time = time.perf_counter()
        inference_A = inference_on_imlist(cfg, model, args.weights_branchA, args.unlabeled_dataset_A_name)
        total_time = time.perf_counter() - start_time
        logger.info("Compute inference on unlabeled dataset A: {:.2f} s".format(total_time))
        start_time = time.perf_counter()
        inference_B = inference_on_imlist(cfg, model, args.weights_branchB, args.unlabeled_dataset_B_name)   
        total_time = time.perf_counter() - start_time
        logger.info("Compute inference on unlabeled dataset B: {:.2f} s".format(total_time))
        logger.info("Pseudolabeling mode: {}".format(pseudolabeling)) 
        if pseudolabeling == 'cbst':
            start_time = time.perf_counter()
            pseudolabels_A, scores_listA = apply_cbst(inference_A, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,len(unlabeled_datasetA),tgt_portion)
            total_time = time.perf_counter() - start_time
            logger.info("CBST on unlabeled dataset A: {:.2f} s".format(total_time))
            start_time = time.perf_counter()
            pseudolabels_B, scores_listB = apply_cbst(inference_B, cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,len(unlabeled_datasetB),tgt_portion)
            total_time = time.perf_counter() - start_time
            logger.info("CBST on unlabeled dataset B: {:.2f} s".format(total_time))
            tgt_portion = min(tgt_portion + TGT_PORT_STEP, MAX_TGT_PORT)
        else:
            raise Exception('unknown pseudolabeling method defined')

        #save pseudolabels
        pseudolabels_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling/pseudolabels')
        create_folder(pseudolabels_path_model_A)
        coloured_pseudolabels_path_model_A = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'pseudolabeling/coloured_pseudolabels')
        create_folder(coloured_pseudolabels_path_model_A)
        pseudolabels_path_model_B = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'pseudolabeling/pseudolabels')
        create_folder(pseudolabels_path_model_B)
        coloured_pseudolabels_path_model_B = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'pseudolabeling/coloured_pseudolabels')
        create_folder(coloured_pseudolabels_path_model_B)
        dataset_A_path = os.path.join(cfg.OUTPUT_DIR,'model_A',str(epoch),'dataset')
        create_folder(dataset_A_path)
        dataset_B_path = os.path.join(cfg.OUTPUT_DIR,'model_B',str(epoch),'dataset')
        create_folder(dataset_B_path)

        logger.info("Collaboration mode: {}".format(collaboration))        
        if collaboration.lower() == 'none': #Self-training for each branch
            # Order pseudolabels by confidences (scores) higher to lower and select number defined to merge with source data
            scores_listA, pseudolabels_A, unlabeled_datasetA = (list(t) for t in zip(*sorted(zip(scores_listA, pseudolabels_A, unlabeled_datasetA),reverse=True)))
            scores_listB, pseudolabels_B, unlabeled_datasetB = (list(t) for t in zip(*sorted(zip(scores_listB, pseudolabels_B, unlabeled_datasetB),reverse=True)))
            # select candidates and save them to add them to the source data
            if len(unlabeled_datasetA) > cfg.PSEUDOLABELING.NUMBER:
                images_txt_A, psedolabels_txt_A = save_pseudolabels(unlabeled_datasetA[:num_selected], pseudolabels_A[:num_selected], scores_listA[:num_selected], pseudolabels_path_model_A, coloured_pseudolabels_path_model_A)
                images_txt_B, psedolabels_txt_B = save_pseudolabels(unlabeled_datasetB[:num_selected], pseudolabels_B[:num_selected], scores_listB[:num_selected], pseudolabels_path_model_B, coloured_pseudolabels_path_model_B)      
            else:
                images_txt_A, psedolabels_txt_A = save_pseudolabels(unlabeled_datasetA, pseudolabels_A, scores_listA, pseudolabels_path_model_A, coloured_pseudolabels_path_model_A)
                images_txt_B, psedolabels_txt_B = save_pseudolabels(unlabeled_datasetB, pseudolabels_B, scores_listB, pseudolabels_path_model_B, coloured_pseudolabels_path_model_B)
        elif collaboration == 'cotraining':
            pass
        else:
            raise Exception('unknown collaboration of models defined')

        # Compute data accumulation procedure
        if accumulation_mode is not None and len(accumulated_selection_imgA) > 0:
            if accumulation_mode.lower() == 'all':
                accumulated_selection_imgA = merge_txts_and_save(accumulated_selection_imgA, images_txt_A, 
                                                                            os.path.join(dataset_A_path,'dataset_img.txt'))
                accumulated_selection_pseudoA = merge_txts_and_save(accumulated_selection_pseudoA, psedolabels_txt_A, 
                                                                            os.path.join(dataset_A_path,'dataset_pseudolabels.txt'))
                accumulated_selection_imgB = merge_txts_and_save(accumulated_selection_imgB, images_txt_B, 
                                                                            os.path.join(dataset_B_path,'dataset_img.txt'))
                accumulated_selection_pseudoB = merge_txts_and_save(accumulated_selection_pseudoB, psedolabels_txt_B, 
                                                                            os.path.join(dataset_B_path,'dataset_pseudolabels.txt'))
        else:
            #No accumulation, only training with new pseudolabels
            accumulated_selection_imgA = images_txt_A
            accumulated_selection_pseudoA = psedolabels_txt_A
            accumulated_selection_imgB = images_txt_B
            accumulated_selection_pseudoB = psedolabels_txt_B


        # create dataloader adding psedolabels to source dataset
        dataset_A_name = cfg.DATASETS.TRAIN_NAME + '_A_' + str(epoch)
        built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT, cfg.DATASETS.TRAIN_GT_TXT, 
                                               dataset_A_name, True, accumulated_selection_imgA, accumulated_selection_pseudoA)
        # Train model A
        logger.info("Training Model A")
        do_train(cfg, model, args.weights_branchA, dataset_A_name, cfg.DATASETS.TEST_NAME, resume=False)


        # create dataloader adding psedolabels to source dataset
        dataset_B_name = cfg.DATASETS.TRAIN_NAME + '_B_' + str(epoch)
        built_custom_dataset(cfg, cfg.DATASETS.TRAIN_IMG_TXT, cfg.DATASETS.TRAIN_GT_TXT, 
                                               dataset_B_name, True, accumulated_selection_imgB, accumulated_selection_pseudoB)

        # Train model B
        logger.info("Training Model B")
        do_train(cfg, model, args.weights_branchB, dataset_B_name, cfg.DATASETS.TEST_NAME, resume=False)

        # delete all datasets registered during epoch
        DatasetCatalog.remove(args.unlabeled_dataset_A_name)
        MetadataCatalog.remove(args.unlabeled_dataset_A_name)
        DatasetCatalog.remove(args.unlabeled_dataset_B_name)
        MetadataCatalog.remove(args.unlabeled_dataset_B_name)
        DatasetCatalog.remove(dataset_A_name)
        MetadataCatalog.remove(dataset_A_name)
        DatasetCatalog.remove(dataset_B_name)
        MetadataCatalog.remove(dataset_B_name)

    '''
    
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model,'eval_only')

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    return do_test(cfg, model, 'final')'''


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
