import cv2
import sys
import numpy as np
import y4m
from cv2.ximgproc import guidedFilter


def process_frame(frame_raw, isheader=False):
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
        ender = b'FRAME\n'
        frame = np.frombuffer(frame_raw[0], dtype='uint8')
        frame = frame.reshape(1080*3//2, 1920)
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
        guided_frame = guidedFilter(frame, frame, 10, 100, -1)
        sys.stdout.buffer.write(guided_frame.tobytes())# + ender)


parser = y4m.Reader(process_frame, verbose=True)
infd = sys.stdin.buffer

with infd as f:
    while True:
        data = f.read(1024)#1920*1080*3//2+6)
        if not data:
            break
        parser.decode(data)
