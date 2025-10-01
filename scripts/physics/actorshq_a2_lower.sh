TRAIN_ITERS=200
ACTOR=2
SEQUENCE=1
TRAIN_FRAME_START=406
TEST_FRAME_START=606
TRAIN_FRAME_NUM=25
TEST_FRAME_NUM=200
VERTS_START_IDX=406
DATADIR="./data"
TRACKDIR="./output/tracking"
MODELDIR="./model"
OUTPUTDIR="./output/phys"
GENDER=male
WANDB_ENTITY=xxxx

########## Train Material Parameters ##########
python train_material_params.py \
--iterations ${TRAIN_ITERS} \
--wandb_name invphys_a${ACTOR}_s${SEQUENCE}_lower \
--save_name a${ACTOR}_s${SEQUENCE}_lower \
--trained_model_path ${TRACKDIR}/a${ACTOR}_s${SEQUENCE}_${VERTS_START_IDX}_200 \
--model_path ${MODELDIR}/a${ACTOR}_s${SEQUENCE} \
--dataset_dir ${DATADIR} \
--output_dir ${OUTPUTDIR} \
--smplx_gender ${GENDER} \
--actor ${ACTOR} \
--sequence ${SEQUENCE} \
--train_frame_start_num ${TRAIN_FRAME_START} ${TRAIN_FRAME_NUM} \
--test_frame_start_num ${TRAIN_FRAME_START} 2 \
--verts_start_idx ${VERTS_START_IDX} \
--init_params_path "" \
--split_idx_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}/split_idx_lower.npz \
--dataset_type actorshq \
--uv_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}/a${ACTOR}s${SEQUENCE}_uv.obj \
--test_camera_index 6 126 \
--grid_size 150 \
--use_wandb \
--visualize \
--wandb_entity ${WANDB_ENTITY}