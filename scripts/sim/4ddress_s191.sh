SUBJECT=191
TRAIN_TAKE=2
TEST_TAKE=8
TRAIN_FRAME_START=21
TEST_FRAME_START=11
TRAIN_FRAME_NUM=12
TEST_FRAME_NUM=100
VERTS_START_IDX=21
DATADIR="./data"
TRACKDIR="./output/tracking"
MODELDIR="./model"
OUTPUTDIR="./output/phys"
GENDER=female

########## Simulation & Rendering ##########
python train_material_params.py \
--save_name s${SUBJECT}_t${TRAIN_TAKE} \
--trained_model_path ${TRACKDIR}/s${SUBJECT}_t${TRAIN_TAKE}_${VERTS_START_IDX}_100 \
--model_path ${MODELDIR}/s${SUBJECT}_t${TRAIN_TAKE} \
--dataset_dir ${DATADIR} \
--output_dir ${OUTPUTDIR} \
--smplx_gender ${GENDER} \
--subject ${SUBJECT} \
--train_take ${TRAIN_TAKE} \
--test_take ${TEST_TAKE} \
--train_frame_start_num ${TRAIN_FRAME_START} 2 \
--test_frame_start_num ${TEST_FRAME_START} ${TEST_FRAME_NUM} \
--verts_start_idx ${VERTS_START_IDX} \
--init_params_path ${OUTPUTDIR}/s${SUBJECT}_t${TRAIN_TAKE}/seed0/best_param_00199.npz \
--split_idx_path ${DATADIR}/s${SUBJECT}_t${TRAIN_TAKE}/split_idx.npz \
--dataset_type 4ddress \
--uv_path ${DATADIR}/s${SUBJECT}_t${TRAIN_TAKE}/mesh_processed.obj \
--test_camera_index 0 \
--run_eval