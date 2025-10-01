ACTOR=6
SEQUENCE=1
TRAIN_FRAME_START=450
TEST_FRAME_START=650
TEST_FRAME_NUM=200
DATADIR="./data"
OUTPUTDIR="./output/phys"

########## Evaluation ##########
python eval.py \
--output_path ${OUTPUTDIR}/a${ACTOR}_s${SEQUENCE}/seed0 \
--mesh_path ${DATADIR}/a${ACTOR}_s${SEQUENCE}/FrameRec000${TRAIN_FRAME_START}.obj \
--data_path ${DATADIR}/ActorsHQ/Actor0${ACTOR}/Sequence${SEQUENCE}/4x/ \
--start_idx ${TEST_FRAME_START} \
--num_timesteps ${TEST_FRAME_NUM} \
--actor ${ACTOR} \
--dataset actorshq