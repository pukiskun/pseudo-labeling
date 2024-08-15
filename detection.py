import os
import cv2
import numpy as np
import glob
import random
import xml.etree.ElementTree as ET
from tensorflow.lite.python.interpreter import Interpreter
import time

def load_labels(label_path):
    with open(label_path, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

def tflite_detect_images(model_path, img_path, lbl_path, min_conf=0.5, num_test_images=10, save_path='output_images', xml_save_path='output_xml'):
    start_time = time.time()

    images = glob.glob(img_path + '/*.jpg') + glob.glob(img_path + '/*.JPG') + glob.glob(img_path + '/*.png') + glob.glob(img_path + '/*.bmp')

    # Load labels
    labels = load_labels(lbl_path)

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    float_input = (input_details[0]['dtype'] == np.float32)
    input_mean = 127.5
    input_std = 127.5

    if len(images) > num_test_images:
        images_to_test = random.sample(images, num_test_images)
    else:
        images_to_test = images

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(xml_save_path, exist_ok=True)

    for image_path in images_to_test:
        image_start_time = time.time()

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imH, imW, _ = image.shape
        image_resized = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_resized, axis=0)

        if float_input:
            input_data = (np.float32(input_data) - input_mean) / input_std

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        boxes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[3]['index'])[0]  # Assuming this output index holds class indices

        detections = []
        for i, score in enumerate(scores):
            if score > min_conf:
                class_id = int(classes[i])
                if class_id < len(labels):
                    object_name = labels[class_id]
                else:
                    object_name = "Unknown"
                ymin = int(max(1, (boxes[i][0] * imH)))
                xmin = int(max(1, (boxes[i][1] * imW)))
                ymax = int(min(imH, (boxes[i][2] * imH)))
                xmax = int(min(imW, (boxes[i][3] * imW)))
                detections.append([object_name, score, xmin, ymin, xmax, ymax])

        create_pascal_voc_xml(image_path, detections, xml_save_path, imW, imH)

        image_end_time = time.time()
        print(f"Time taken for {os.path.basename(image_path)}: {image_end_time - image_start_time:.2f} seconds")

    total_time = time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")

def create_pascal_voc_xml(image_path, detections, output_folder, width, height):
    filename = os.path.basename(image_path)
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = os.path.basename(output_folder)
    ET.SubElement(annotation, "filename").text = filename
    ET.SubElement(annotation, "path").text = image_path
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3" 

    if detections:
        for detection in detections:
            obj = ET.SubElement(annotation, "object")
            object_name, confidence, xmin, ymin, xmax, ymax = detection
            ET.SubElement(obj, "name").text = object_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"

            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".xml")
    tree.write(output_path, encoding='utf-8', xml_declaration=False)

def draw_bounding_boxes(image_folder, xml_folder, show_in_gui=False):
    for xml_file in os.listdir(xml_folder):
        if not xml_file.endswith('.xml'):
            continue
        
        xml_path = os.path.join(xml_folder, xml_file)
        image_path = os.path.join(image_folder, os.path.splitext(xml_file)[0] + ".jpg")
        
        if not os.path.exists(image_path):
            continue

        image = cv2.imread(image_path)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall('object'):
            name = obj.find('name').text
            xmin = int(obj.find('bndbox/xmin').text)
            ymin = int(obj.find('bndbox/ymin').text)
            xmax = int(obj.find('bndbox/xmax').text)
            ymax = int(obj.find('bndbox/ymax').text)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if show_in_gui:
            cv2.imshow('Labeled Image', image)
            cv2.waitKey(0)  # Wait until a key is pressed
            cv2.destroyAllWindows()
        else:
            output_path = os.path.join(image_folder, os.path.splitext(xml_file)[0] + "_labeled.jpg")
            cv2.imwrite(output_path, image)
            print(f"Bounding boxes drawn on {image_path} and saved to {output_path}")

if __name__ == "__main__":
    MODEL_PATH = "F:\\TA Dataset\\custom_model_lite\\detect.tflite"
    IMAGE_FOLDER = "F:\\TA Dataset\\raw\\images"
    LABEL_PATH = "F:\\TA Dataset\\custom_model_lite\\labelmap.txt"
    OUTPUT_IMAGE_FOLDER = "F:\\TA Dataset\\raw\\out"
    OUTPUT_XML_FOLDER = "F:\\TA Dataset\\raw\\images"
    MIN_CONF_THRESHOLD = 0.1
    NUM_IMAGES_TO_TEST = 1378

    tflite_detect_images(MODEL_PATH, IMAGE_FOLDER, LABEL_PATH, MIN_CONF_THRESHOLD, NUM_IMAGES_TO_TEST, OUTPUT_IMAGE_FOLDER, OUTPUT_XML_FOLDER)
    draw_bounding_boxes(IMAGE_FOLDER, OUTPUT_XML_FOLDER, show_in_gui=False)
