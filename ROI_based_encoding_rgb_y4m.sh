video_name='PDSTSP_653--20230129160608216--PG--2012'
video_name_output_crf31='/internal-demo3/bingyu/gf_1080p_crf31_mp4/'$video_name'_rawvideo_bgr24.mp4'
video_name_output='/internal-demo3/bingyu/gf_1080p_crf31_merge_mp4/'$video_name'_rawvideo_bgr24.mp4'
resolution=`ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 /internal-demo3/bingyu/raw_mxf/$video_name.mxf`
fps=`ffprobe -v error -select_streams v -of default=noprint_wrappers=1:nokey=1 -show_entries stream=r_frame_rate /internal-demo3/bingyu/raw_mxf/$video_name.mxf`
fps_num=(${fps//// })
start=`date +%s`
ffmpeg -i /internal-demo3/bingyu/raw_mxf/$video_name.mxf  -filter_complex "[0:v]yadif" -pix_fmt yuv420p -f yuv4mpegpipe - | python pipe_non_ROI_process.py | ffmpeg -f rawvideo -pixel_format bgr24 -video_size $resolution -framerate ${fps_num[0]} -i - -pix_fmt yuv420p -an -write_tmcd 0 -muxdelay 0 -c:v libx265 -b_strategy 2 -err_detect compliant -ssim -psnr -tag:v hvc1 -g 50 -keyint_min 50 -profile:v main  -movflags +faststart -sws_flags +accurate_rnd+full_chroma_inp+full_chroma_int -coder 1 -flags +loop -bidir_refine 1 -qmin 4 -qdiff 4 -qmax 69 -crf 31 -x265-params keyint=50:min-keyint=50:rc-lookahead=50:level-idc=4:vbv-bufsize=2160:vbv-maxrate=1620:scenecut=0:scenecut-bias=0:open-gop=0:pbratio=1.3:me=umh:qcomp=0.5:ipratio=1.4:b-adapt=1:ref=6:bframes=4:frame-threads=1:pools=10:hrd-concat=1 -color_range tv -colorspace bt709 -color_trc bt709 -color_primaries bt709 -pix_fmt yuv420p -aspect 16:9  -vsync 0 $video_name_output_crf31 -y
end1=`date +%s`
ffmpeg -i /internal-demo3/bingyu/raw_mxf/$video_name.mxf  -filter_complex "[0:v]yadif" -pix_fmt yuv420p -f yuv4mpegpipe - | python pipe_merge_frame_y4m.py -video_name $video_name | ffmpeg -f rawvideo -pixel_format bgr24 -video_size $resolution -framerate ${fps_num[0]} -i - -an -write_tmcd 0 -muxdelay 0 -c:v libx265 -b_strategy 2 -err_detect compliant -ssim -psnr -tag:v hvc1 -g 50 -keyint_min 50 -profile:v main  -movflags +faststart -sws_flags +accurate_rnd+full_chroma_inp+full_chroma_int -coder 1 -flags +loop -bidir_refine 1 -qmin 4 -qdiff 4 -qmax 69 -crf 28 -x265-params keyint=50:min-keyint=50:rc-lookahead=50:level-idc=4:vbv-bufsize=2160:vbv-maxrate=1620:scenecut=0:scenecut-bias=0:open-gop=0:pbratio=1.3:me=umh:qcomp=0.5:ipratio=1.4:b-adapt=1:ref=6:bframes=4:frame-threads=1:pools=10:hrd-concat=1 -color_range tv -colorspace bt709 -color_trc bt709 -color_primaries bt709 -pix_fmt yuv420p -aspect 16:9  -vsync 0 $video_name_output -y
end2=`date +%s`
time1=`echo $start $end1 | awk '{print $2-$1}'`
echo $time1
time2=`echo $start $end2 | awk '{print $2-$1}'`
echo $time2
# 4723(only this process)+13502(two processes)
