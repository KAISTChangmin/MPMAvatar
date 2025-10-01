TRAIN_ITERS=30000
SUBJECT=191
TAKE=2
START_IDX=21
NUM_FRAMES=100
DATADIR="./data"
TRACKDIR="./output/tracking"
MODELDIR="./model"

########## Train Appearance ##########
python train_appearance.py \
-m ${MODELDIR}/s${SUBJECT}_t${TAKE} \
--iterations ${TRAIN_ITERS} \
--dataset_dir ${DATADIR} \
--uv_path ${DATADIR}/s${SUBJECT}_t${TAKE}/mesh_processed.obj \
--subject ${SUBJECT} \
--train_take ${TAKE} \
--test_take ${TAKE} \
--train_frame_start_num ${START_IDX} ${NUM_FRAMES} \
--test_frame_start_num ${START_IDX} ${NUM_FRAMES} \
--trained_model_path ${TRACKDIR}/s${SUBJECT}_t${TAKE}_${START_IDX}_${NUM_FRAMES} \
--test_camera_index 0 \
--dataset_type 4ddress