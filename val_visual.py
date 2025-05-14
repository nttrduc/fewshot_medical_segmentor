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
from PIL import Image, ImageDraw # Thêm thư viện PIL

from models.grid_proto_fewshot import FewShotSeg # Đảm bảo model này tồn tại và đúng

# Giả sử các dataloader khác vẫn cần thiết và đường dẫn đúng
from dataloaders.dev_customized_med import med_fewshot_val
from dataloaders.ManualAnnoDatasetv2 import ManualAnnoDataset
from dataloaders.GenericSuperDatasetv2 import SuperpixelDataset
from dataloaders.dataset_utils import DATASET_INFO, get_normalize_op
from dataloaders.niftiio import convert_to_sitk

from util.metric import Metric # Có thể không cần nếu chỉ lưu ảnh

from config_ssl_upload import ex # Đảm bảo file config này tồn tại

import tqdm # tqdm có thể không cần thiết nếu bạn không dùng progress bar trong vòng lặp này
import SimpleITK as sitk
# from torchvision.utils import make_grid # Có thể không cần

# config pre-trained model caching path
os.environ['TORCH_HOME'] = "./pretrained_model"

# --- Phần tùy chỉnh thêm ---
OUTPUT_VIS_DIR = "segmentation_visualizations_comparison" # Đổi tên thư mục output
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

# Định nghĩa màu sắc cho từng cơ quan và cho các loại mask
ORGAN_COLORS = [ # Màu dùng để tạo thư mục, không trực tiếp vẽ lên ảnh với màu này nữa
    "RedOrgan", "GreenOrgan", "BlueOrgan", "YellowOrgan"
]
GT_COLOR_OVERLAY = (0, 255, 255, 100) # Cyan (R,G,B,Alpha) cho ground truth overlay
ALP_PRED_COLOR_OVERLAY = (255, 0, 255, 100) # Magenta cho ALP prediction overlay
QSM_PRED_COLOR_OVERLAY = (255, 165, 0, 100) # Orange cho QSM prediction overlay

def tensor_to_pil(tensor_img, normalize_per_channel=False):
    """Chuyển tensor ảnh (C, H, W) hoặc (H,W) sang ảnh PIL (H, W, C hoặc H, W)."""
    if tensor_img.is_cuda:
        tensor_img = tensor_img.cpu()

    img_np = tensor_img.numpy()

    if img_np.ndim == 3: # (C,H,W)
        if normalize_per_channel:
            for c in range(img_np.shape[0]):
                channel_min = img_np[c, :, :].min()
                channel_max = img_np[c, :, :].max()
                img_np[c, :, :] = (img_np[c, :, :] - channel_min) / (channel_max - channel_min + 1e-5) * 255
        else: # Normalize chung
            img_min = img_np.min()
            img_max = img_np.max()
            img_np = (img_np - img_min) / (img_max - img_min + 1e-5) * 255
        img_np = np.transpose(img_np, (1, 2, 0)) # Chuyển sang (H,W,C)
    elif img_np.ndim == 2: # (H,W) - Grayscale
        img_min = img_np.min()
        img_max = img_np.max()
        img_np = (img_np - img_min) / (img_max - img_min + 1e-5) * 255
    else:
        raise ValueError(f"Tensor input có số chiều không hợp lệ: {img_np.ndim}")

    img_np = img_np.astype(np.uint8)

    if img_np.ndim == 2 or img_np.shape[2] == 1:
        return Image.fromarray(img_np.squeeze(), 'L')
    elif img_np.shape[2] == 3:
        return Image.fromarray(img_np, 'RGB')
    else:
        raise ValueError(f"Ảnh numpy có số kênh không hợp lệ: {img_np.shape}")


def draw_mask_on_image(image_pil, mask_np, color):
    """Vẽ mask (numpy array H,W, giá trị 0 hoặc 1) lên ảnh PIL."""
    if image_pil.mode != 'RGBA':
        image_pil = image_pil.convert("RGBA")

    if mask_np.dtype != bool: # Đảm bảo mask là boolean
        mask_np = mask_np.astype(bool)

    overlay = Image.new('RGBA', image_pil.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)

    # Vẽ các pixel của mask lên overlay
    for y in range(mask_np.shape[0]):
        for x in range(mask_np.shape[1]):
            if mask_np[y, x]: # Nếu pixel này thuộc mask
                draw.point((x, y), fill=color)

    combined_image = Image.alpha_composite(image_pil, overlay)
    return combined_image.convert("RGB") # Chuyển lại RGB để lưu


@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        observer_dir = _run.observers[0].dir
        os.makedirs(f'{observer_dir}/interm_preds_viz', exist_ok=True) # Thư mục riêng cho viz
        # ... (phần lưu source code nếu cần) ...
        pass

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    # THAY ĐỔI: Đường dẫn cụ thể đến weight của QSM và ALP
    # Ví dụ: snapshots_dir = _config["snapshots_dir"]
    snapshots_base_dir = _config.get("snapshots_dir", "/root/ducnt/fewshot_medical_segmentor/exps/myexp_MIDDLE_0/mySSL_train_SABS_Superpix_lbgroup0_scale_MIDDLE_vfold0_SABS_Superpix_sets_0_1shot/18/snapshots")

    qsm_weight_filename = _config.get("qsm_weight_file", "QSM_model.pth") # Lấy từ config hoặc mặc định
    alp_weight_filename = _config.get("alp_weight_file", "ALP_model.pth") # Lấy từ config hoặc mặc định

    qsm_weight_path = os.path.join(snapshots_base_dir, qsm_weight_filename)
    alp_weight_path = os.path.join(snapshots_base_dir, alp_weight_filename)


    if not os.path.exists(qsm_weight_path):
        _log.error(f"QSM weight not found at: {qsm_weight_path}")
        return 1 # Trả về mã lỗi
    if not os.path.exists(alp_weight_path):
        _log.error(f"ALP weight not found at: {alp_weight_path}")
        return 1 # Trả về mã lỗi

    model_weights_paths = {
        "QSM": qsm_weight_path,
        "ALP": alp_weight_path
    }

    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'SABS_Superpix':
        baseset_name = 'SABS'
        max_label = 13
    elif data_name == 'CHAOST2_Superpix': # Đã thêm CHAOST2
        baseset_name = 'CHAOST2'
        max_label = 4 # Max label cho CHAOST2
    else:
        raise ValueError(f'Dataset: {data_name} not found')

    test_labels = _config.get("specific_test_labels", [1, 2, 3, 4])
    if len(test_labels) > len(ORGAN_COLORS) and _config.get("use_organ_color_names_for_folders", True):
        _log.warning(f"Số lượng test_labels ({len(test_labels)}) nhiều hơn số tên màu định nghĩa ({len(ORGAN_COLORS)}). Một số thư mục cơ quan sẽ dùng index.")

    _log.info(f'###### Unseen labels to visualize: {test_labels} ######')

    if baseset_name == 'SABS':
        tr_parent = SuperpixelDataset(
            which_dataset=baseset_name,
            base_dir=_config['path'][data_name]['data_dir'],
            idx_split=_config['eval_fold'],
            mode='train',
            min_fg=str(_config["min_fg_data"]),
            transforms=None,
            nsup=_config['task']['n_shots'],
            scan_per_load=_config['scan_per_load'],
            exclude_list=_config["exclude_cls_list"],
            superpix_scale=_config["superpix_scale"],
            fix_length=_config["max_iters_per_load"] if (data_name == 'C0_Superpix') or (
                        data_name == 'CHAOST2_Superpix') else None
        )
        norm_func = tr_parent.norm_func
    elif baseset_name == 'CHAOST2':
        # CHAOST2 là MR, cần normalize khác SABS (CT)
        # Giả sử bạn có logic tương tự SuperpixelDataset cho CHAOST2 hoặc dùng normalize mặc định cho MR
        # Nếu không có tr_parent cho CHAOST2, bạn cần norm_func từ nguồn khác
        norm_func = get_normalize_op(modality='MR', fids=None) # Hoặc logic normalize cụ thể của bạn
    else:
        norm_func = get_normalize_op(modality='MR', fids=None)


    te_dataset, te_parent = med_fewshot_val(
        dataset_name=baseset_name,
        base_dir=_config['path'][baseset_name]['data_dir'],
        idx_split=_config['eval_fold'],
        scan_per_load=_config.get('scan_per_load_val', -1), # Cho phép config riêng cho val
        act_labels=test_labels,
        npart=_config['task']['npart'],
        nsup=_config['task']['n_shots'],
        extern_normalize_func=norm_func
    )

    testloader = DataLoader(
        te_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=_config.get('num_workers_val', 1), # Cho phép config
        pin_memory=False,
        drop_last=False
    )
    _log.info('###### Starting Visualization Generation ######')

    processed_organ_count = 0

    for organ_idx_loop, curr_lb in enumerate(test_labels):
        if processed_organ_count >= 4 and _config.get("limit_to_4_organs_viz", True) : # Chỉ xử lý tối đa 4 cơ quan
            _log.info("Đã xử lý đủ 4 cơ quan. Dừng.")
            break

        _log.info(f"--- Processing Organ Label: {curr_lb} (Loop index: {organ_idx_loop}) ---")

        if _config.get("use_organ_color_names_for_folders", True):
            folder_name_suffix = ORGAN_COLORS[organ_idx_loop % len(ORGAN_COLORS)]
            organ_output_dir = os.path.join(OUTPUT_VIS_DIR, f"organ_{curr_lb}_{folder_name_suffix}")
        else:
            organ_output_dir = os.path.join(OUTPUT_VIS_DIR, f"organ_{curr_lb}")

        os.makedirs(organ_output_dir, exist_ok=True)

        te_dataset.set_curr_cls(curr_lb)

        support_batched_info = te_parent.get_support(
            curr_class=curr_lb,
            class_idx=[curr_lb],
            scan_idx=_config.get("support_idx_viz", _config["support_idx"]), # Cho phép config support_idx riêng cho viz
            npart=_config['task']['npart']
        )

        # --- DEBUGGING SUPPORT STRUCTURE ---
        _log.info(f"Support Info Keys: {list(support_batched_info.keys())}")
        _log.info(f"Type of support_batched_info['support_images']: {type(support_batched_info.get('support_images'))}")
        if isinstance(support_batched_info.get('support_images'), list):
            _log.info(f"Length of support_batched_info['support_images']: {len(support_batched_info['support_images'])}")
            # ... (Thêm các log chi tiết hơn nếu cần, tương tự như lần trước) ...
        # --- END DEBUGGING SUPPORT STRUCTURE ---

        valid_support_structure = True
        support_image_tensor = None
        if not isinstance(support_batched_info.get('support_images'), list) or not support_batched_info['support_images']:
            valid_support_structure = False
        elif not isinstance(support_batched_info['support_images'][0], list) or not support_batched_info['support_images'][0]: # part
            valid_support_structure = False
        elif not isinstance(support_batched_info['support_images'][0][0], list) or not support_batched_info['support_images'][0][0]: # shot
            valid_support_structure = False
        elif not torch.is_tensor(support_batched_info['support_images'][0][0][0]):
            valid_support_structure = False
        else:
            # Giả sử 1 shot, 1 part, 1 way
            support_image_tensor = support_batched_info['support_images'][0][0][0].clone().detach()

        if not valid_support_structure:
            _log.error(f"Cấu trúc của support_batched_info['support_images'] không hợp lệ hoặc rỗng cho label {curr_lb}. Bỏ qua.")
            continue

        valid_mask_structure = True
        support_gt_mask_tensor = None
        if not isinstance(support_batched_info.get('support_mask'), list) or not support_batched_info['support_mask']:
            valid_mask_structure = False
        elif not isinstance(support_batched_info['support_mask'][0], list) or not support_batched_info['support_mask'][0]: # part
            valid_mask_structure = False
        elif not isinstance(support_batched_info['support_mask'][0][0], dict): # shot (là dict)
            valid_mask_structure = False
        else:
            first_shot_support_mask_dict = support_batched_info['support_mask'][0][0]
            mask_key_found = False
            for key_candidate in [f'fg_mask', 'mask', f'fg_gt', 'gt']: # Thử nhiều key phổ biến
                if key_candidate in first_shot_support_mask_dict and torch.is_tensor(first_shot_support_mask_dict[key_candidate]):
                    support_gt_mask_tensor = first_shot_support_mask_dict[key_candidate].float().clone().detach()
                    mask_key_found = True
                    break
            if not mask_key_found:
                valid_mask_structure = False
                _log.error(f"Không tìm thấy key mask hợp lệ (fg_mask, mask, fg_gt, gt) trong support_mask dict: {first_shot_support_mask_dict.keys()}")


        if not valid_mask_structure or support_gt_mask_tensor is None:
            _log.error(f"Cấu trúc của support_batched_info['support_mask'] không hợp lệ, rỗng, hoặc không tìm thấy key mask cho label {curr_lb}. Bỏ qua.")
            continue

        support_pil = tensor_to_pil(support_image_tensor.squeeze()) # Squeeze nếu channel = 1
        support_gt_mask_np = support_gt_mask_tensor.squeeze().cpu().numpy()
        support_with_gt_pil = draw_mask_on_image(support_pil.copy(), support_gt_mask_np, GT_COLOR_OVERLAY)
        support_with_gt_pil.save(os.path.join(organ_output_dir, f"0_support_organ_{curr_lb}_gt.png"))
        _log.info(f"Đã lưu ảnh support cho cơ quan {curr_lb}")

        query_sample_batched = None
        query_image_tensor_for_vis = None # Dùng để vẽ ảnh query gốc
        query_gt_mask_for_vis = None    # Dùng để vẽ ảnh query gốc
        query_image_for_model = None    # Dùng cho input model (có thể khác shape/type)
        found_query = False

        for sample_idx, sample in enumerate(testloader):
            _scan_id = sample["scan_id"][0]
            if _scan_id in te_parent.potential_support_sid:
                continue

            current_query_gt_mask_np = sample['label'].squeeze().cpu().numpy()
            # Kiểm tra xem mask có ít nhất một vài pixel foreground không (thay vì chỉ 1)
            if np.sum(current_query_gt_mask_np == 1) > _config.get("min_query_fg_pixels_viz", 10): # Có thể config
                query_sample_batched = sample
                query_image_tensor_for_vis = query_sample_batched['image'].squeeze(0).clone().detach() # (C,H,W)
                query_gt_mask_for_vis = query_sample_batched['label'].squeeze().clone().detach()    # (H,W)
                query_image_for_model = query_sample_batched['image'].clone().detach() # (B,C,H,W) cho model
                found_query = True
                _log.info(f"Đã tìm thấy query slice cho cơ quan {curr_lb} từ scan {_scan_id}, slice_z {sample['z_id'][0]}, sample_idx {sample_idx} trong testloader.")
                break
            if sample_idx > _config.get("max_query_search_viz", 200): # Giới hạn số lượng query tìm kiếm
                _log.warning(f"Đã tìm kiếm quá {_config.get('max_query_search_viz', 200)} query cho organ {curr_lb} mà không thấy. Bỏ qua.")
                break

        if not found_query:
            _log.warning(f"Không tìm thấy query slice phù hợp cho cơ quan {curr_lb}. Bỏ qua cơ quan này.")
            continue

        query_pil = tensor_to_pil(query_image_tensor_for_vis)
        query_gt_mask_np = query_gt_mask_for_vis.cpu().numpy()
        query_with_gt_pil = draw_mask_on_image(query_pil.copy(), query_gt_mask_np, GT_COLOR_OVERLAY)
        query_with_gt_pil.save(os.path.join(organ_output_dir, f"1_query_organ_{curr_lb}_gt.png"))
        _log.info(f"Đã lưu ảnh query (ground truth) cho cơ quan {curr_lb}")

        # Chuẩn bị support cho model
        # Cần shape: way(1) x shot(1) x [B(1) x C x H x W]
        sup_img_part_model = [[support_image_tensor.cuda().unsqueeze(0)]]
        # Cần shape: way(1) x shot(1) x [B(1) x 1 x H x W] hoặc (B, H, W) tùy model chấp nhận
        # Nếu support_gt_mask_tensor là (H,W), unsqueeze 2 lần
        if support_gt_mask_tensor.ndim == 2:
            sup_fgm_part_tensor = support_gt_mask_tensor.unsqueeze(0).unsqueeze(0) # B=1, C=1, H, W
        elif support_gt_mask_tensor.ndim == 3 and support_gt_mask_tensor.shape[0] == 1: # Đã là (1,H,W)
            sup_fgm_part_tensor = support_gt_mask_tensor.unsqueeze(0) # B=1, C=1, H, W
        else: # Đã là (C,H,W) hoặc (B,C,H,W)
            sup_fgm_part_tensor = support_gt_mask_tensor
        if sup_fgm_part_tensor.ndim == 3: # Nếu mới chỉ là (C,H,W)
            sup_fgm_part_tensor = sup_fgm_part_tensor.unsqueeze(0) # Thêm batch dim

        sup_fgm_part_model = [[sup_fgm_part_tensor.cuda()]]
        sup_bgm_part_model = [[(1 - sup_fgm_part_tensor.cuda())]] # Tạo bg mask

        for model_name, weight_path in model_weights_paths.items():
            _log.info(f"--- Inferring with {model_name} for Organ {curr_lb} ---")
            backbone_cfg = _config['model'].get('backbone', 'resnet50') # Default
            # Cố gắng lấy backbone từ tên file nếu có quy tắc
            # Ví dụ: QSM_resnet101_epoch50.pth -> resnet101
            # Đây là ví dụ đơn giản, bạn có thể cần regex hoặc logic phức tạp hơn
            fn_parts = os.path.basename(weight_path).lower().split('_')
            potential_bb_names = ["resnet50", "resnet101", "vgg16", "mobilenet", "mitb0", "mitb1"] # Thêm các backbone có thể có
            found_bb_in_fn = None
            for part in fn_parts:
                if part in potential_bb_names:
                    found_bb_in_fn = part
                    # Xử lý trường hợp đặc biệt như mit_b0
                    if part == "mit" and len(fn_parts) > fn_parts.index(part) + 1 and fn_parts[fn_parts.index(part)+1].startswith("b"):
                         found_bb_in_fn = f"mit_{fn_parts[fn_parts.index(part)+1]}"
                    break
            if found_bb_in_fn:
                backbone_to_use = found_bb_in_fn
                _log.info(f"Sử dụng backbone '{backbone_to_use}' từ tên file weight cho {model_name}.")
            else:
                backbone_to_use = backbone_cfg
                _log.info(f"Sử dụng backbone '{backbone_to_use}' từ config cho {model_name}.")


            model = FewShotSeg(pretrained_path=weight_path, cfg=_config['model'], backbone=backbone_to_use)
            model = model.cuda()
            model.eval()

            with torch.no_grad():
                query_pred_logits, _, _, _ = model(sup_img_part_model,
                                                 sup_fgm_part_model,
                                                 sup_bgm_part_model,
                                                 [query_image_for_model.cuda()], # List of query images
                                                 isval=True,
                                                 val_wsize=_config.get("val_wsize", -1))

                # Xử lý output của model
                # Giả sử output là (B, NumClasses, H, W)
                # Nếu model của bạn được huấn luyện để output 2 class (BG, FG) cho few-shot
                if query_pred_logits.shape[1] == 2:
                    predicted_mask_tensor = query_pred_logits.argmax(dim=1) # (B, H, W), lấy class FG
                # Nếu model output 1 class (đã là FG score)
                elif query_pred_logits.shape[1] == 1:
                    predicted_mask_tensor = (torch.sigmoid(query_pred_logits) > 0.5).long().squeeze(1) # (B,H,W)
                else: # Nhiều class, và bạn cần xác định class nào tương ứng với curr_lb
                      # Điều này phức tạp hơn và phụ thuộc vào thiết kế model
                      # Trong trường hợp đơn giản nhất của few-shot, model thường chỉ output 1 hoặc 2 class
                    _log.warning(f"Model {model_name} output {query_pred_logits.shape[1]} classes. Mặc định lấy argmax.")
                    predicted_mask_tensor = query_pred_logits.argmax(dim=1)


            predicted_mask_np = predicted_mask_tensor.squeeze().cpu().numpy() # (H,W)

            pred_color = ALP_PRED_COLOR_OVERLAY if model_name == "ALP" else QSM_PRED_COLOR_OVERLAY
            query_with_pred_pil = draw_mask_on_image(query_pil.copy(), predicted_mask_np, pred_color)

            filename_index = 2 if model_name == "ALP" else 3
            query_with_pred_pil.save(os.path.join(organ_output_dir, f"{filename_index}_query_organ_{curr_lb}_pred_{model_name}.png"))
            _log.info(f"Đã lưu ảnh query (predicted by {model_name}) cho cơ quan {curr_lb}")

            del model
            torch.cuda.empty_cache()

        processed_organ_count += 1 # Đã xử lý xong một cơ quan
        _log.info(f"--- Đã hoàn thành xử lý cho cơ quan {curr_lb} ---")

    _log.info('###### End of Visualization Generation ######')
    return 0 # Trả về 0 nếu thành công
