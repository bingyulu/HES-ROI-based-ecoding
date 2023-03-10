import cv2
import sys
import numpy as np
import y4m
from cv2.ximgproc import guidedFilter
import argparse
import os
from PIL import Image


def get_noblur_frame(frame_noblur_face):
    frame_noblur_face = np.array(frame_noblur_face)
    shot_cls_dir = root_dir + "/shot_cls_result/"+video_name[:-1]+"_scale.txt"
    shot_dir = root_dir + "/shot_cls_result/"+video_name[:-1]+"_0.1.txt"
    
    f_cls = open(shot_cls_dir)
    cls_label = f_cls.readline()
    labels = []
    frame_ind = []
    while cls_label:
        labels.append(int(cls_label.split(',')[0]))
        cls_label = f_cls.readline()
    f_cls.close()
    
    f_shot = open(shot_dir, encoding='utf-8')
    shot_ = f_shot.readline()
    for i in range(len(labels)):
        shot_ = shot_.strip('\n')
        shot_list = shot_.split('\t')
        frame_range = np.arange(int(shot_list[0]), int(shot_list[1])+1)
        noblur_shot_intersect = np.intersect1d(frame_noblur_face, frame_range)
        noblur_face_ratio = len(noblur_shot_intersect)/len(frame_range)
        if labels[i] != 0:
            frame_ind.extend(list(frame_range))
        elif noblur_face_ratio > 0.5:
            frame_ind.extend(list(frame_range))
        shot_ = f_shot.readline()
    f_shot.close()
    
    return frame_ind


parser = argparse.ArgumentParser()
parser.add_argument("-video_name", type=str, default='YRKKH_3963--20230129192718416--PG--2012')
args = parser.parse_args() 
video_name = args.video_name+'/'
root_dir = '/internal-demo3/bingyu'
frame_noblur_face = []
frame_noblur = get_noblur_frame(frame_noblur_face)

cap = cv2.VideoCapture('/internal-demo3/bingyu/gf_1080p_crf31_mp4/'+video_name[:-1]+'_rawvideo_bgr24.mp4')
cap_raw = cv2.VideoCapture('/internal-demo3/bingyu/raw_mxf/'+video_name[:-1]+'.mxf')
frame_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
global frame_ind
frame_ind = 1


def process_frame(frame_raw, isheader=False):
    global frame_ind
    if isheader:
        # header = b''
        # for ele in frame_raw:
        #     if header == b'':
        #         header = ele
        #     else:
        #         header = header + b' ' + ele
        # header = header + b'\n'
        # header = b'YUV4MPEG2 W1920 H1080 F25:1 Ip A1:1 C444 XYSCSS=444\n'
        sys.stdout.flush()
        # sys.stdout.buffer.write(header)
    else:
        frame = np.frombuffer(frame_raw[0], dtype='uint8')
        frame = frame.reshape(1080*3//2, 1920)
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        
        _, blured_bg = cap.read()
        
        # not process full&long shots
        if frame_ind in frame_noblur:
            sys.stdout.flush()
            sys.stdout.buffer.write(frame.tobytes())
            frame_ind += 1
            return
        
        if frame_ind<1001:
            humanseg = cv2.imread(os.path.join(root_dir+'/matting_masks/'+video_name[:-1], str(frame_ind-1).zfill(4)+'.png'))
        else:
            humanseg = cv2.imread(os.path.join(root_dir+'/matting_masks/'+video_name[:-1], str(frame_ind-1)+'.png'))
        humanseg = cv2.cvtColor(humanseg,cv2.COLOR_RGB2GRAY)

        # thr=100 for humanseg masks, thr=0 for matting masks
        _, humanseg_binary = cv2.threshold(humanseg, 0, 255, cv2.THRESH_BINARY)

        merge_mask = cv2.GaussianBlur(humanseg_binary,(201,201),0)
        merge_mask = Image.fromarray(merge_mask)
        merge_mask = merge_mask.convert('L')
        
        blured_bg = Image.fromarray(cv2.cvtColor(blured_bg,cv2.COLOR_BGR2RGB))
        frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))

        blured_bg = blured_bg.convert('RGBA')
        frame = frame.convert('RGBA')

        blured_bg = Image.composite(frame, blured_bg, mask=merge_mask)

        blured_bg = blured_bg.convert('RGB')
        blured_bg = np.array(blured_bg, dtype=np.uint8)
        blured_bg = cv2.cvtColor(blured_bg, cv2.COLOR_RGB2BGR)

        sys.stdout.flush()
        sys.stdout.buffer.write(blured_bg.tobytes())
        frame_ind += 1
        

parser = y4m.Reader(process_frame, verbose=True)
infd = sys.stdin.buffer

with infd as f:
    while True:
        data = f.read(1024)#1920*1080*3//2+6)
        if not data:
            break
        parser.decode(data)
        
