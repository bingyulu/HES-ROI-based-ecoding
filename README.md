# HES-ROI-based-ecoding
ROI based encoding project in Python.

The ROI based encoding approach includes four parts: ROI mask generation, non-ROI compression, ROI-based frame merging. First, we use the human video matting model to generate ROI masks of all frames. The code can be found at https://github.com/PeterL1n/RobustVideoMatting/blob/master/README_zh_Hans.md.
The masks should be saved in the matting_masks/$video_name$/ folder.

Then we should prepare the shot classification results using the shot boundary detection and shot scale classification model. Remember to download weights to the shot_boundary_detection folder before. More detailed explanation can be found in shot_boundary_detection/README.md.

Command for detecting shot boundary:
```
python shot_boundary_detection/inference.py --result_dir ./shot_cls_result/ --video ./1080p_crf28_mp4/$video_name$  --threshold 0.1
```
Command for classify shots:
```
python shot_boundary_detection/shot_scale_classification.py --shot_dir ./shot_cls_result/ --video_dir ./1080p_crf28_mp4/$video_name$ --frame_dir ./frames/1080p_crf28_frames/
```
The non-ROI compression and ROI-based frame merging step are done by the ROI_based_encoding_rgb_y4m.sh. Before running, you should chanage the variable video_name in the script. Besides, you may want to change the variable video_name_output_crf31 and video_name_output to your video output path. The root_dir in pipe_merge_frame_y4m.py should also be changed  according to your video output path.
```
bash ROI_based_encoding_rgb_y4m.sh
```
Finally the encoded video can be found in video_name_output.
