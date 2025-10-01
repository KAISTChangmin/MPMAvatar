ACTOR=1
SEQUENCE=1
START_IDX=460
NUM_FRAMES=200
DATADIR="../data"
TRACKDIR="../output/tracking"
CLOTH_NAME="cloth_sim.obj"
GENDER=neutral
WANDB_ENTITY=xxxx

python train_mesh_lbs_actorshq.py \
--save_name a${ACTOR}_s${SEQUENCE}_${START_IDX}_${NUM_FRAMES} \
--seq a${ACTOR}_s${SEQUENCE} \
--start_idx ${START_IDX} \
--num_frames ${NUM_FRAMES} \
--smplx_gender ${GENDER} \
--obj_name FrameRec000${START_IDX}.obj \
--cloth_name ${CLOTH_NAME} \
--data_path ${DATADIR}/ActorsHQ/Actor0${ACTOR}/Sequence${SEQUENCE} \
--wandb \
--wandb_entity ${WANDB_ENTITY} \
--wandb_name track_a${ACTOR}_s${SEQUENCE}_${START_IDX}_${NUM_FRAMES}

python ../blender/add_uv_actorshq.py \
--uv_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}/a${ACTOR}s${SEQUENCE}_uv.obj \
--output_path ${TRACKDIR}/a${ACTOR}_s${SEQUENCE}_${START_IDX}_${NUM_FRAMES}

blender -b -P ../blender/bake.py -- \
--output_path ${TRACKDIR}/a${ACTOR}_s${SEQUENCE}_${START_IDX}_${NUM_FRAMES}

python lbs_weights_inpainting_actorshq.py \
--smplx_gender ${GENDER} \
--src_mesh_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}/smplx_fitted/000${START_IDX}.obj \
--src_param_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}/smplx_fitted/000${START_IDX}.pth \
--target_mesh_path ${TRACKDIR}/a${ACTOR}_s${SEQUENCE}_${START_IDX}_${NUM_FRAMES}/mesh_cloth_${START_IDX}.obj \
--output_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}

python split_garments.py \
--mesh_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}/FrameRec000${START_IDX}.obj \
--cloth_obj ${DATADIR}/a${ACTOR}_s${SEQUENCE}/cloth_sim.obj \
--fix_v None \
--iteration 20 \
--filename ${DATADIR}/a${ACTOR}_s${SEQUENCE}/split_idx.npz