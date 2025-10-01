SUBJECT=185
TAKE=1
START_IDX=11
NUM_FRAMES=100
DATADIR="../data"
TRACKDIR="../output/tracking"
GENDER=female
WANDB_ENTITY=xxxx

python train_mesh_lbs_4ddress.py \
--save_name s${SUBJECT}_t${TAKE}_${START_IDX}_${NUM_FRAMES} \
--seq s${SUBJECT}_t${TAKE} \
--start_idx ${START_IDX} \
--num_frames ${NUM_FRAMES} \
--labels 3 4 \
--data_path ${DATADIR}/4D-DRESS/00${SUBJECT}_Inner/Inner/Take${TAKE} \
--wandb \
--wandb_entity ${WANDB_ENTITY} \
--wandb_name track_s${SUBJECT}_t${TAKE}_${START_IDX}_${NUM_FRAMES}

python ../blender/add_uv_4ddress.py \
--uv_path ${DATADIR}/s${SUBJECT}_t${TAKE}/mesh_processed.obj \
--output_path ${TRACKDIR}/s${SUBJECT}_t${TAKE}_${START_IDX}_${NUM_FRAMES}

blender -b -P ../blender/bake.py -- \
--output_path ${TRACKDIR}/s${SUBJECT}_t${TAKE}_${START_IDX}_${NUM_FRAMES}

python lbs_weights_inpainting_4ddress.py \
--smplx_gender ${GENDER} \
--src_mesh_path ${DATADIR}/4D-DRESS/00${SUBJECT}_Inner/Inner/Take${TAKE}/SMPLX/mesh-f000${START_IDX}_smplx.ply \
--src_param_path ${DATADIR}/4D-DRESS/00${SUBJECT}_Inner/Inner/Take${TAKE}/SMPLX/mesh-f000${START_IDX}_smplx.pkl \
--target_mesh_path ${TRACKDIR}/s${SUBJECT}_t${TAKE}_${START_IDX}_${NUM_FRAMES}/mesh_cloth_${START_IDX}.obj \
--output_path ${DATADIR}/s${SUBJECT}_t${TAKE}

python split_garments.py \
--mesh_path ${DATADIR}/s${SUBJECT}_t${TAKE}/mesh_processed.obj \
--cloth_npz ${DATADIR}/s${SUBJECT}_t${TAKE}/cloth_vertices.npz \
--labels 3 \
--fix_v None \
--iteration 0 \
--filename ${DATADIR}/s${SUBJECT}_t${TAKE}/split_idx_upper.npz

python split_garments.py \
--mesh_path ${DATADIR}/s${SUBJECT}_t${TAKE}/mesh_processed.obj \
--cloth_npz ${DATADIR}/s${SUBJECT}_t${TAKE}/cloth_vertices.npz \
--labels 4 \
--fix_v ${DATADIR}/s${SUBJECT}_t${TAKE}/fix_v.npy \
--iteration 20 \
--filename ${DATADIR}/s${SUBJECT}_t${TAKE}/split_idx_lower.npz