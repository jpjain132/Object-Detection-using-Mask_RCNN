import cv2 as cv
import argparse
import numpy as np
import os.path
import sys
import random
import time
import signal

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
maskThreshold = 0.3  # Mask threshold

parser = argparse.ArgumentParser(description='Use this script to run Mask-RCNN object detection and segmentation')
parser.add_argument('--image', help='Path to image file')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument("--device", default="cpu", help="Device to inference on")
args = parser.parse_args()

# Accuracy metrics
total_frames = 0
total_detections = 0
total_confidence = 0
frames_with_detections = 0
total_inference_time = 0

fps_list = []

# Draw the predicted bounding box, colorize and show the mask on the image
def drawBox(frame, classId, conf, left, top, right, bottom, classMask):
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
    classMask = cv.resize(classMask, (right - left + 1, bottom - top + 1))
    mask = (classMask > maskThreshold)
    roi = frame[top:bottom+1, left:right+1][mask]
    colorIndex = random.randint(0, len(colors)-1)
    color = colors[colorIndex]
    frame[top:bottom+1, left:right+1][mask] = ([0.3*color[0], 0.3*color[1], 0.3*color[2]] + 0.7 * roi).astype(np.uint8)
    mask = mask.astype(np.uint8)
    contours, hierarchy = cv.findContours(mask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame[top:bottom+1, left:right+1], contours, -1, color, 3, cv.LINE_8, hierarchy, 100)

# For each frame, extract the bounding box and mask for each detected object
def postprocess(boxes, masks):
    global total_detections, total_confidence, frames_with_detections
    numClasses = masks.shape[1]
    numDetections = boxes.shape[2]
    frameH = frame.shape[0]
    frameW = frame.shape[1]
    detections_in_this_frame = 0
    for i in range(numDetections):
        box = boxes[0, 0, i]
        mask = masks[i]
        score = box[2]
        if score > confThreshold:
            classId = int(box[1])
            left = int(frameW * box[3])
            top = int(frameH * box[4])
            right = int(frameW * box[5])
            bottom = int(frameH * box[6])
            left = max(0, min(left, frameW - 1))
            top = max(0, min(top, frameH - 1))
            right = max(0, min(right, frameW - 1))
            bottom = max(0, min(bottom, frameH - 1))
            classMask = mask[classId]
            drawBox(frame, classId, score, left, top, right, bottom, classMask)
            total_detections += 1
            total_confidence += score
            detections_in_this_frame += 1
    if detections_in_this_frame > 0:
        frames_with_detections += 1

classesFile = "mscoco_labels.names";
classes = None
with open(classesFile, 'rt') as f:
   classes = f.read().rstrip('\n').split('\n')

textGraph = "./mask_rcnn_inception_v2_coco_2018_01_28.pbtxt";
modelWeights = "./mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb";
net = cv.dnn.readNetFromTensorflow(modelWeights, textGraph);

if args.device == "cpu":
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

colorsFile = "colors.txt";
with open(colorsFile, 'rt') as f:
    colorsStr = f.read().rstrip('\n').split('\n')
colors = []
for i in range(len(colorsStr)):
    rgb = colorsStr[i].split(' ')
    color = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])])
    colors.append(color)

winName = 'Mask-RCNN Object detection and Segmentation in OpenCV'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = "mask_rcnn_out_py.avi"
if (args.image):
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_mask_rcnn_out_py.jpg'
elif (args.video):
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_mask_rcnn_out_py.avi'
else:
    cap = cv.VideoCapture(0)

if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 28, (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

def print_metrics():
    print("\n==== Detection Stats Summary ====")
    print(f"Total frames processed     : {total_frames}")
    print(f"Total detections           : {total_detections}")
    print(f"Average detections/frame   : {total_detections/total_frames if total_frames else 0:.2f}")
    print(f"Frames with detections     : {frames_with_detections}")
    print(f"Detection presence rate    : {(frames_with_detections/total_frames)*100 if total_frames else 0:.2f}%")
    print(f"Average confidence score   : {total_confidence/total_detections if total_detections else 0:.2f}")
    print(f"Average FPS                : {np.mean(fps_list) if fps_list else 0:.2f}")
    print("=================================\n")

try:
    while cv.waitKey(1) < 0:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            break
        start_time = time.time()
        total_frames += 1
        blob = cv.dnn.blobFromImage(frame, swapRB=True, crop=False)
        net.setInput(blob)
        boxes, masks = net.forward(['detection_out_final', 'detection_masks'])
        postprocess(boxes, masks)
        t, _ = net.getPerfProfile()
        inf_time = t * 1000.0 / cv.getTickFrequency()
        total_inference_time += inf_time
        fps_list.append(1000.0 / inf_time if inf_time > 0 else 0)
        label = 'Inference time: %0.0f ms' % inf_time
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
        if (args.image):
            cv.imwrite(outputFile, frame.astype(np.uint8));
        else:
            vid_writer.write(frame.astype(np.uint8))
        cv.imshow(winName, frame)

except KeyboardInterrupt:
    print("\n\n[INFO] Ctrl+C detected. Finalizing and showing metrics...")
    cap.release()
    if not args.image:
        vid_writer.release()
    print_metrics()
    cv.destroyAllWindows()
    sys.exit(0)
