# HES-ROI-based-ecoding
ROI based encoding project in Python.

The ROI based encoding approach includes four parts: ROI mask generation, non-ROI compression, ROI-based frame merging. First, we use the human video matting model to generate ROI masks of all frames. The code can be found at https://github.com/PeterL1n/RobustVideoMatting/blob/master/README_zh_Hans.md.
The masks should be saved in the matting_masks/$video_name$/ folder.

Then we should prepare the shot classification results using the shot boundary detection and shot scale classification model.

Command for detecting shot boundary:
python shot_boundary_detection/inference.py --result_dir ./shot_cls_result/ --video ./1080p_crf28_mp4/$video_name$  --threshold 0.1

Command for classify shots:
python shot_boundary_detection/shot_scale_classification.py --shot_dir ./shot_cls_result/ --video_dir ./1080p_crf28_mp4/$video_name$ --frame_dir ./frames/1080p_crf28_frames/

The non-ROI compression and ROI-based frame merging step are done by the ROI_based_encoding_rgb_y4m.sh. Before running, we should chanage the variable video_name in the script, the output path video_name_output can also be changed in the script if you want.
bash ROI_based_encoding_rgb_y4m.sh

Finally the encoded video can be found in video_name_output.
