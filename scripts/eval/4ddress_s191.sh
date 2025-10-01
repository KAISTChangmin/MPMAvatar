SUBJECT=191
TRAIN_TAKE=2
TEST_TAKE=8
TEST_FRAME_START=11
TEST_FRAME_NUM=100
DATADIR="./data"
OUTPUTDIR="./output/phys"

########## Evaluation ##########
python eval.py \
--output_path ${OUTPUTDIR}/s${SUBJECT}_t${TRAIN_TAKE}/seed0 \
--mesh_path ${DATADIR}/s${SUBJECT}_t${TRAIN_TAKE}/mesh_processed.obj \
--data_path ${DATADIR}/4D-DRESS/00${SUBJECT}_Inner/Inner/Take${TEST_TAKE} \
--start_idx ${TEST_FRAME_START} \
--num_timesteps ${TEST_FRAME_NUM} \
--dataset 4ddress