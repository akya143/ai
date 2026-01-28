import numpy as np
import os
import tensorflow as tf
import cv2
MODEL_NAME = 'ssd_mobilenet_v2_coco_2018_03_29'
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
PATH_TO_LABELS = "C:\MSC-IT SEM-2\Computer vision\ssd_mobilenet_v2_coco_2018_03_29"
category_index = {}
with open(PATH_TO_LABELS, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if 'id:' in line:
            id_index = int(line.strip().split(':')[1])
        if 'display_name:' in line:
            name = line.strip().split(':')[1].strip().strip('"')
            category_index[id_index] = {'name': name}
def detect_objects(image):
    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_expanded = np.expand_dims(image, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_expanded})
            for i in range(len(scores[0])):
                if scores[0][i] > 0.5:  # Adjust confidence threshold as needed
                    class_id = int(classes[0][i])
                    class_name = category_index[class_id]['name']
                    score = float(scores[0][i])
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    (left, right, top, bottom) = (xmin * image.shape[1], xmax * image.shape[1], ymin * image.shape[0], ymax * image.shape[0])
                    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                    cv2.putText(image, '{}: {:.2f}'.format(class_name, score), (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
input_image = cv2.imread("C:\MSC-IT SEM-2\Computer vision\IMG.jpg")
output_image = detect_objects(input_image)
cv2.imshow('Object Detection', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
