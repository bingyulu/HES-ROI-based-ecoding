set -e
eval "$(conda shell.bash hook)"

NAME=shot_detection
conda activate $NAME

VIDEO_DIR=${1:-/home/ubuntu/data/test_video}
SHOT_DIR=shot_txt
KEYF_DIR=keyf
# Run Shot Detection on videos in $VIDEO_DIR
python inference.py --result_dir $SHOT_DIR --video $VIDEO_DIR --threshold 0.1

# Extract Keyframes
# For scene detection: 3 keyframes per shot
python extract_keyframe.py --video_dir $VIDEO_DIR --shot_dir $SHOT_DIR --keyf_dir ${KEYF_DIR}_scene --mode scene --num_keyf 3
# For video tagging, 1 keyframe per second, minimum 3 keyframes per shot
python extract_keyframe.py --video_dir $VIDEO_DIR --shot_dir $SHOT_DIR --keyf_dir $KEYF_DIR --mode tagging --interval 1 --min_num_keyf 3
