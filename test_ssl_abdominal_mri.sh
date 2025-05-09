#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# === Config ===
PROTO_GRID=8
DATASET='CHAOST2_Superpix'
CPT="myexp"
ALL_EV=(0)
ALL_SCALE=("MIDDLE")
LABEL_SETS=0
EXCLU='[1,2,3,4]'
SUPP_ID='[4]'
NWORKER=4
NSTEP=100100
MAX_ITER=1000
SNAPSHOT_INTERVAL=25000
DECAY=0.95
SEED='1234'
WEIGHT_DIR="/root/ducnt/fewshot_medical_segmentor/exps/myexperiments_MIDDLE_0/mySSL_train_CHAOST2_Superpix_lbgroup0_scale_MIDDLE_vfold0_CHAOST2_Superpix_sets_0_1shot/62/snapshots"

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



# #!/bin/bash
# # train a model to segment abdominal MRI 
# GPUID1=0
# export CUDA_VISIBLE_DEVICES=$GPUID1

# ####### Shared configs ######
# PROTO_GRID=8 # using 32 / 8 = 4, 4-by-4 prototype pooling window during training
# CPT="myexp"
# DATASET='CHAOST2_Superpix'
# NWORKER=4

# ALL_EV=( 0) # 5-fold cross validation (0, 1, 2, 3, 4)
# ALL_SCALE=( "MIDDLE") # config of pseudolabels

# ### Use L/R kidney as testing classes
# LABEL_SETS=0 
# EXCLU='[2,3]' # setting 2: excluding kidneies in training set to test generalization capability even though they are unlabeled. Use [] for setting 1 by Roy et al.

# ### Use Liver and spleen as testing classes
# # LABEL_SETS=1 
# # EXCLU='[1,4]' 

# ###### Training configs (irrelavent in testing) ######
# NSTEP=100100
# DECAY=0.95

# MAX_ITER=1000 # defines the size of an epoch
# SNAPSHOT_INTERVAL=25000 # interval for saving snapshot
# SEED='1234'

# ###### Validation configs ######
# SUPP_ID='[4]'  # using the additionally loaded scan as support

# echo ===================================

# for EVAL_FOLD in "${ALL_EV[@]}"
# do
#     for SUPERPIX_SCALE in "${ALL_SCALE[@]}"
#     do
#     PREFIX="test_vfold${EVAL_FOLD}"
#     echo $PREFIX
#     LOGDIR="./exps/${CPT}"

#     if [ ! -d $LOGDIR ]
#     then
#         mkdir $LOGDIR
#     fi

#     RELOAD_PATH='' # path to the reloaded model

#     python3 validation.py with \
#     'modelname=dlfcn_res101' \
#     'usealign=True' \
#     'optim_type=sgd' \
#     reload_model_path=$RELOAD_PATH \
#     num_workers=$NWORKER \
#     scan_per_load=-1 \
#     label_sets=$LABEL_SETS \
#     'use_wce=True' \
#     exp_prefix=$PREFIX \
#     'clsname=grid_proto' \
#     n_steps=$NSTEP \
#     exclude_cls_list=$EXCLU \
#     eval_fold=$EVAL_FOLD \
#     dataset=$DATASET \
#     proto_grid_size=$PROTO_GRID \
#     max_iters_per_load=$MAX_ITER \
#     min_fg_data=1 seed=$SEED \
#     save_snapshot_every=$SNAPSHOT_INTERVAL \
#     superpix_scale=$SUPERPIX_SCALE \
#     lr_step_gamma=$DECAY \
#     path.log_dir=$LOGDIR \
#     support_idx=$SUPP_ID
#     done
# done