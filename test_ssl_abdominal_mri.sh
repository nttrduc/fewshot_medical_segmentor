#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# === Config ===
PROTO_GRID=8
DATASET='CHAOST2_Superpix'
CPT="myexp"
ALL_EV=(0)
ALL_SCALE=("MIDDLE")
LABEL_SETS=0
EXCLU='[2,3]'
SUPP_ID='[4]'
NWORKER=4
NSTEP=100100
MAX_ITER=1000
SNAPSHOT_INTERVAL=25000
DECAY=0.95
SEED='1234'
WEIGHT_DIR="/root/ducnt/fewshot_medical_segmentor/exps/myexperiments_MIDDLE_0/mySSL_train_CHAOST2_Superpix_lbgroup0_scale_MIDDLE_vfold0_CHAOST2_Superpix_sets_0_1shot/121/snapshots/"

# === Start ===
for EVAL_FOLD in "${ALL_EV[@]}"; do
  for SUPERPIX_SCALE in "${ALL_SCALE[@]}"; do
    PREFIX="test_vfold${EVAL_FOLD}"
    LOGDIR="./exps/${CPT}"
    mkdir -p "$LOGDIR"

    for RELOAD_PATH in "$WEIGHT_DIR"/*.pth; do
      if [ ! -f "$RELOAD_PATH" ]; then
        echo "⚠️ Không tìm thấy file .pth nào trong $WEIGHT_DIR"
        exit 1
      fi

      echo "========================================="
      echo "▶ Testing with model: $RELOAD_PATH"
      echo "========================================="

      python3 validation.py with \
        modelname=dlfcn_res101 \
        usealign=True \
        optim_type=sgd \
        reload_model_path=$RELOAD_PATH \
        num_workers=$NWORKER \
        scan_per_load=-1 \
        label_sets=$LABEL_SETS \
        use_wce=True \
        exp_prefix=$PREFIX \
        clsname=grid_proto \
        n_steps=$NSTEP \
        exclude_cls_list=$EXCLU \
        eval_fold=$EVAL_FOLD \
        dataset=$DATASET \
        proto_grid_size=$PROTO_GRID \
        max_iters_per_load=$MAX_ITER \
        min_fg_data=1 \
        seed=$SEED \
        save_snapshot_every=$SNAPSHOT_INTERVAL \
        superpix_scale=$SUPERPIX_SCALE \
        lr_step_gamma=$DECAY \
        path.log_dir=$LOGDIR \
        support_idx=$SUPP_ID

    done
  done
done
