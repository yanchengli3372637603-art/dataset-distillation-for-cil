import os
import os.path
import sys
import logging
import copy
import time
import torch
import numpy as np
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def _parse_device_spec(device_spec):
    raw = str(device_spec).replace('，', ',')
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if len(parts) == 0:
        parts = ['0']
    train_part = parts[0]
    posteval_part = parts[1] if len(parts) > 1 else None
    return train_part, posteval_part, parts


def train(args):
    seed_list = copy.deepcopy(args['seed'])
    train_device_raw, posteval_device_raw, all_devices_raw = _parse_device_spec(copy.deepcopy(args['device']))

    for seed in seed_list:
        args['seed'] = seed
        args['device'] = [train_device_raw]
        if posteval_device_raw is not None:
            args['posteval_device'] = posteval_device_raw
        args['all_cli_devices'] = all_devices_raw
        _train(args)

def _train(args):
    logdir = 'logs/{}/{}_tasks'.format(args['dataset'], args['total_sessions'])
    args['logdir'] = logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfilename = os.path.join(logdir, '{}_ca:{}_msc:{}_cc:{}_rank:{}_{}_{}_{}-{}'.format(args['seed'], args["ca"], args["msc"], args["cc"], args['rank'], args["lora_type"], args['model_name'], args['optim'], args['lrate']))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    print(logfilename)
    _set_random(args)
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args['dataset'], 
        args['shuffle'], 
        args['seed'], 
        args['init_cls'], 
        args['increment'], 
        args
    )
    model = factory.get_model(args['model_name'], args)


    cnn_curve, cnn_curve_with_task, nme_curve, cnn_curve_task = {'top1': []}, {'top1': []}, {'top1': []}, {'top1': []}
    for task_id in range(data_manager.nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        time_start = time.time()
        model.incremental_train(data_manager)
        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
        time_start = time.time()
        cnn_accy, cnn_accy_with_task, nme_accy, cnn_accy_task = model.eval_task()
        time_end = time.time()
        logging.info('Time:{}'.format(time_end - time_start))
        if hasattr(model, 'run_post_eval_hooks'):
            hook_start = time.time()
            model.run_post_eval_hooks()
            hook_end = time.time()
            logging.info('Post-eval hook time:{}'.format(hook_end - hook_start))
        model.after_task()

        logging.info('CNN: {}'.format(cnn_accy['grouped']))
        cnn_curve['top1'].append(cnn_accy['top1'])
        cnn_curve_with_task['top1'].append(cnn_accy_with_task['top1'])
        cnn_curve_task['top1'].append(cnn_accy_task)
        logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        logging.info('CNN top1 with task curve: {}'.format(cnn_curve_with_task['top1']))
        logging.info('CNN top1 task curve: {}'.format(cnn_curve_task['top1']))

        if task_id > 0:
            diagonal = np.diag(model.acc_matrix)
            forgetting = np.mean((np.max(model.acc_matrix, axis=1) -
                                model.acc_matrix[:, task_id])[:task_id])
            backward = np.mean((model.acc_matrix[:, task_id] - diagonal)[:task_id])

            result_str = "Forgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
            logging.info(result_str)

    logging.info('Accuracy Matrix: \n {}'.format(model.acc_matrix.T.round(2)))
    logging.info('Average Accuracy: {}'.format(np.mean(cnn_curve['top1'])))
    logging.info('Last Accuracy: {}'.format(cnn_curve['top1'][-1]))

def _set_device(args):
    device_type = args['device']
    gpus = []

    for device in device_type:
        device_str = str(device).strip()
        if device_str in ('-1', 'cpu'):
            resolved = torch.device('cpu')
        elif device_str.startswith('cuda:'):
            resolved = torch.device(device_str)
        else:
            resolved = torch.device('cuda:{}'.format(device_str))
        gpus.append(resolved)

    args['device'] = gpus


def _set_random(args):
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
    if 'posteval_device' in args:
        logging.info('posteval_device(resolved from --device if provided): {}'.format(args['posteval_device']))

