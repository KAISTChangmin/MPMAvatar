TRAIN_ITERS=30000
ACTOR=2
SEQUENCE=1
START_IDX=406
NUM_FRAMES=200
DATADIR="./data"
TRACKDIR="./output/tracking"
MODELDIR="./model"

########## Train Appearance ##########
python train_appearance.py \
-m ${MODELDIR}/a${ACTOR}_s${SEQUENCE} \
--iterations ${TRAIN_ITERS} \
--dataset_dir ${DATADIR} \
--uv_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}/a${ACTOR}s${SEQUENCE}_uv.obj \
--actor ${ACTOR} \
--sequence ${SEQUENCE} \
--train_frame_start_num ${START_IDX} ${NUM_FRAMES} \
--test_frame_start_num ${START_IDX} ${NUM_FRAMES} \
--trained_model_path ${TRACKDIR}/a${ACTOR}_s${SEQUENCE}_${START_IDX}_${NUM_FRAMES} \
--test_camera_index 6 126 \
--dataset_type actorshq