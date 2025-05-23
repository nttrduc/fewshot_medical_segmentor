"""
Validation script
"""
import csv
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
import numpy as np
import time
from models.grid_proto_fewshot import FewShotSeg

from dataloaders.dev_customized_med import med_fewshot_val
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk

from util.metric import Metric

from config_ssl_upload import ex

import tqdm
import SimpleITK as sitk
from torchvision.utils import make_grid

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/interm_preds', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)
    
    snapshots_dir = "/root/ducnt/fewshot_medical_segmentor/exps/myexperiments_MIDDLE_1/mySSL_train_CHAOST2_Superpix_lbgroup1_scale_MIDDLE_vfold0_CHAOST2_Superpix_sets_1_1shot/5/snapshots"

    weight_files = [f for f in os.listdir(snapshots_dir) if f.endswith('.pth')]


    _log.info('###### Load data ######')
    ### Training set
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'C0_Superpix':
        raise NotImplementedError
        baseset_name = 'C0'
        max_label = 3
    elif data_name == 'CHAOST2_Superpix':
        baseset_name = 'CHAOST2'
        max_label = 4
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] - DATASET_INFO[baseset_name]['LABEL_GROUP'][_config["label_sets"]]
    # test_labels = DATASET_INFO[baseset_name]['LABEL_GROUP']['pa_all'] 

    ### Transforms for data augmentation
    te_transforms = None

    assert _config['scan_per_load'] < 0 # by default we load the entire dataset directly

    _log.info(f'###### Labels excluded in training : {[lb for lb in _config["exclude_cls_list"]]} ######')
    _log.info(f'###### Unseen labels evaluated in testing: {[lb for lb in test_labels]} ######')

    if baseset_name == 'SABS': # for CT we need to know statistics of 
        tr_parent = SuperpixelDataset( # base dataset
            which_dataset = baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split = _config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]), # dummy entry for superpixel dataset
            transforms=None,
            nsup = _config['task']['n_shots'],
            scan_per_load = _config['scan_per_load'],
            exclude_list = _config["exclude_cls_list"],
            superpix_scale = _config["superpix_scale"],
            fix_length = _config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (data_name == 'CHAOST2_Superpix') else None
        )
        norm_func = tr_parent.norm_func
    else:
        norm_func = get_normalize_op(modality = 'MR', fids = None)


    te_dataset, te_parent = med_fewshot_val(
        dataset_name = baseset_name,
        base_dir=_config['path'][baseset_name]['data_dir'],
        idx_split = _config['eval_fold'],
        scan_per_load = _config['scan_per_load'],
        act_labels=test_labels,
        npart = _config['task']['npart'],
        nsup = _config['task']['n_shots'],
        extern_normalize_func = norm_func
    )

    ### dataloaders
    testloader = DataLoader(
        te_dataset,
        batch_size = 1,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        drop_last=False
    )

    _log.info('###### Set validation nodes ######')
    mar_val_metric_node_qsm = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])
    mar_val_metric_node_alp = Metric(max_label=max_label, n_scans= len(te_dataset.dataset.pid_curr_load) - _config['task']['n_shots'])

    _log.info('###### Starting validation ######')


    for weight_file in weight_files:
        pretrained_path = os.path.join(snapshots_dir, weight_file)
        print("="*10)
        print("Starting validation for weight file:", pretrained_path )
        # _log.info(f'###### Reload model {_config["reload_model_path"]} ######')
        model = FewShotSeg(pretrained_path = pretrained_path, cfg=_config['model'], backbone="mobile")
        model = model.cuda()
        model.eval()
        mar_val_metric_node_qsm.reset()
        mar_val_metric_node_alp.reset()

        with torch.no_grad():
            save_pred_buffer = {} # indexed by class

            _log.info('###### Start validation ######')
            start_time = time.time()
            for curr_lb in test_labels:
                te_dataset.set_curr_cls(curr_lb)
                support_batched = te_parent.get_support(curr_class = curr_lb, class_idx = [curr_lb], scan_idx = _config["support_idx"], npart=_config['task']['npart'])

                # way(1 for now) x part x shot x 3 x H x W] #
                support_images = [[shot.cuda() for shot in way]
                                    for way in support_batched['support_images']] # way x part x [shot x C x H x W]
                suffix = 'mask'
                support_fg_mask = [[shot[f'fg_{suffix}'].float().cuda() for shot in way]
                                    for way in support_batched['support_mask']]
                support_bg_mask = [[shot[f'bg_{suffix}'].float().cuda() for shot in way]
                                    for way in support_batched['support_mask']]

                curr_scan_count = -1 # counting for current scan
                _lb_buffer = {} # indexed by scan

                last_qpart = 0 # used as indicator for adding result to buffer

                for sample_batched in testloader:

                    _scan_id = sample_batched["scan_id"][0] # we assume batch size for query is 1
                    if _scan_id in te_parent.potential_support_sid: # skip the support scan, don't include that to query
                        continue
                    if sample_batched["is_start"]:
                        ii = 0
                        curr_scan_count += 1
                        _scan_id = sample_batched["scan_id"][0]
                        outsize = te_dataset.dataset.info_by_scan[_scan_id]["array_size"]
                        outsize = (256, 256, outsize[0]) # original image read by itk: Z, H, W, in prediction we use H, W, Z
                        _pred = np.zeros( outsize )
                        _pred.fill(np.nan)

                    q_part = sample_batched["part_assign"] # the chunck of query, for assignment with support
                    query_images = [sample_batched['image'].cuda()]
                    query_labels = torch.cat([ sample_batched['label'].cuda()], dim=0)

                    # [way, [part, [shot x C x H x W]]] ->
                    sup_img_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_images[0][q_part]]]   # way(1) x shot x [B(1) x C x H x W]
                    sup_fgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_fg_mask[0][q_part]]]
                    sup_bgm_part = [[shot_tensor.unsqueeze(0) for shot_tensor in support_bg_mask[0][q_part]]]

                    query_pred_alp, query_pred_qsm, _, _, _ = model( sup_img_part , sup_fgm_part, sup_bgm_part, query_images, isval = True, val_wsize = _config["val_wsize"] )

                    query_pred_qsm = np.array(query_pred_qsm.argmax(dim=1)[0].cpu())
                    _pred[..., ii] = query_pred_qsm.copy()

                    if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                        mar_val_metric_node_qsm.record(query_pred_qsm, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) 
                    else:
                        pass
                    query_pred_alp = np.array(query_pred_alp.argmax(dim=1)[0].cpu())
                    _pred[..., ii] = query_pred_alp.copy()

                    if (sample_batched["z_id"] - sample_batched["z_max"] <= _config['z_margin']) and (sample_batched["z_id"] - sample_batched["z_min"] >= -1 * _config['z_margin']):
                        mar_val_metric_node_alp.record(query_pred_alp, np.array(query_labels[0].cpu()), labels=[curr_lb], n_scan=curr_scan_count) 
                    else:
                        pass

                    ii += 1
                    # now check data format
                    if sample_batched["is_end"]:
                        if _config['dataset'] != 'C0':
                            _lb_buffer[_scan_id] = _pred.transpose(2,0,1) # H, W, Z -> to Z H W
                        else:
                            _lb_buffer[_scan_id] = _pred

                save_pred_buffer[str(curr_lb)] = _lb_buffer

            # ### save results
            # for curr_lb, _preds in save_pred_buffer.items():
            #     for _scan_id, _pred in _preds.items():
            #         _pred *= float(curr_lb)
            #         itk_pred = convert_to_sitk(_pred, te_dataset.dataset.info_by_scan[_scan_id])
            #         fid = os.path.join(f'{_run.observers[0].dir}/interm_preds', f'scan_{_scan_id}_label_{curr_lb}.nii.gz')
            #         sitk.WriteImage(itk_pred, fid, True)

        # del save_pred_buffer

            # del sample_batched, support_images, support_bg_mask, query_images, query_labels, query_pred

            # compute dice scores by scan
            # --- QSM ---
            m_classDice_qsm, _, m_meanDice_qsm, _, m_rawDice_qsm = mar_val_metric_node_qsm.get_mDice(labels=sorted(test_labels), n_scan=None, give_raw=True)
            m_classPrec_qsm, _, m_meanPrec_qsm, _, m_classRec_qsm, _, m_meanRec_qsm, _, m_rawPrec_qsm, m_rawRec_qsm = mar_val_metric_node_qsm.get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw=True)
            mar_val_metric_node_qsm.reset()

            # --- ALP ---
            m_classDice_alp, _, m_meanDice_alp, _, m_rawDice_alp = mar_val_metric_node_alp.get_mDice(labels=sorted(test_labels), n_scan=None, give_raw=True)
            m_classPrec_alp, _, m_meanPrec_alp, _, m_classRec_alp, _, m_meanRec_alp, _, m_rawPrec_alp, m_rawRec_alp = mar_val_metric_node_alp.get_mPrecRecall(labels=sorted(test_labels), n_scan=None, give_raw=True)
            mar_val_metric_node_alp.reset()

            # # write validation result to log file
            # _run.log_scalar('mar_val_batches_classDice', m_classDice.tolist())
            # _run.log_scalar('mar_val_batches_meanDice', m_meanDice.tolist())
            # _run.log_scalar('mar_val_batches_rawDice', m_rawDice.tolist())

            # _run.log_scalar('mar_val_batches_classPrec', m_classPrec.tolist())
            # _run.log_scalar('mar_val_batches_meanPrec', m_meanPrec.tolist())
            # _run.log_scalar('mar_val_batches_rawPrec', m_rawPrec.tolist())

            # _run.log_scalar('mar_val_batches_classRec', m_classRec.tolist())
            # _run.log_scalar('mar_val_al_batches_meanRec', m_meanRec.tolist())
            # _run.log_scalar('mar_val_al_batches_rawRec', m_rawRec.tolist())

            # _log.info(f'mar_val batches classDice: {m_classDice}')
            # _log.info(f'mar_val batches meanDice: {m_meanDice}')

            # _log.info(f'mar_val batches classPrec: {m_classPrec}')
            # _log.info(f'mar_val batches meanPrec: {m_meanPrec}')

            # _log.info(f'mar_val batches classRec: {m_classRec}')
            # _log.info(f'mar_val batches meanRec: {m_meanRec}')

            # print("============ ============")

            _log.info(f'End of validation')
            
            
            csv_path = 'backbone_validation_report.csv'
            write_header = not os.path.exists(csv_path)

            # Tạo header
            if write_header:
                header = (
                    ['Information'] +
                    [f'Dice_organ{i+1}' for i in range(len(m_classDice_qsm))] + ['Dice_Mean'] +
                    [f'Prec_organ{i+1}' for i in range(len(m_classPrec_qsm))] + ['Prec_Mean'] +
                    [f'Rec_organ{i+1}' for i in range(len(m_classRec_qsm))] + ['Rec_Mean'] +
                    ['path']
                )

            # Dòng dữ liệu QSM
            row_qsm = [f"{data_name}_QSM_0.7_7"]
            row_qsm += [round(x.item(), 4) for x in m_classDice_qsm] + [round(m_meanDice_qsm.item(), 4)]
            row_qsm += [round(x.item(), 4) for x in m_classPrec_qsm] + [round(m_meanPrec_qsm.item(), 4)]
            row_qsm += [round(x.item(), 4) for x in m_classRec_qsm] + [round(m_meanRec_qsm.item(), 4)]
            row_qsm += [pretrained_path]

            # Dòng dữ liệu ALP
            row_alp = [f"{data_name}_ALP_0.7_0.7"]
            row_alp += [round(x.item(), 4) for x in m_classDice_alp] + [round(m_meanDice_alp.item(), 4)]
            row_alp += [round(x.item(), 4) for x in m_classPrec_alp] + [round(m_meanPrec_alp.item(), 4)]
            row_alp += [round(x.item(), 4) for x in m_classRec_alp] + [round(m_meanRec_alp.item(), 4)]
            row_alp += [pretrained_path]

            # Ghi vào file CSV
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(header)
                writer.writerow([])         # <<== Dòng trống
                writer.writerow(row_alp)
                writer.writerow(row_qsm)

        
    return 1


    