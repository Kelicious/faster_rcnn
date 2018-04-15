import cv2
import os
from xml.etree import ElementTree
from shapes import Box, GroundTruthBox, Image, Metadata

IMAGES_DIR = 'JPEGImages'
ANNOTATIONS_DIR = 'Annotations'
IMAGESETS_DIR = 'ImageSets/Main'

VOC_CLASS_MAPPING = {
    'aeroplane': 0,
    'bicycle': 1,
    'bird': 2,
    'boat': 3,
    'bottle': 4,
    'bus': 5,
    'car': 6,
    'cat': 7,
    'chair': 8,
    'cow': 9,
    'diningtable': 10,
    'dog': 11,
    'horse': 12,
    'motorbike': 13,
    'person': 14,
    'pottedplant': 15,
    'sheep': 16,
    'sofa': 17,
    'train': 18,
    'tvmonitor': 19,
    'bg': 20
}

KITTI_CLASS_MAPPING = {
    'car': 0,
    'person': 1,
    'Cyclist': 2,
    'DontCare': 3,
    'Misc': 4,
    'Person_sitting': 5,
    'Tram': 6,
    'Truck': 7,
    'Van': 8,
    'bg': 9
}

'''
Types to use:

img: raw image
img_metadata: {
    width: int,
    height: int,
    bboxes: list<bbox>
}

bbox: {
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    name: string
}

'''


def extract_img_metadata(base_path, img_num):
    images_base = os.path.join(base_path, IMAGES_DIR)
    annotations_base = os.path.join(base_path, ANNOTATIONS_DIR)
    annotations_path = os.path.join(annotations_base, img_num + '.xml')

    if not os.path.exists(annotations_path):
        # hack for KITTI test data since there are no annotation files and we need annotation files to load the data
        filename = img_num + '.png'
        image_path = os.path.join(images_base, filename)
        raw_img = cv2.imread(image_path)
        height, width, depth = raw_img.shape
        root_node = ElementTree.Element('annotation')
        filename_node = ElementTree.Element('filename')
        filename_node.text = filename
        root_node.append(filename_node)
        size_node = ElementTree.Element('size')
        width_node = ElementTree.Element('width')
        width_node.text = str(width)
        height_node = ElementTree.Element('height')
        height_node.text = str(height)
        depth_node = ElementTree.Element('depth')
        depth_node.text = str(depth)
        size_node.append(width_node)
        size_node.append(height_node)
        size_node.append(depth_node)
        root_node.append(size_node)

        with open(annotations_path, 'w+') as f:
            to_write = str(ElementTree.tostring(root_node), 'utf-8')
            f.write(to_write)

    xml = ElementTree.parse(annotations_path)
    annotation = xml.getroot()
    image_path = os.path.join(images_base, annotation.find('filename').text)
    size = annotation.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    gt_boxes = []

    for object in annotation.findall('object'):
        name = object.find('name').text
        bndbox = object.find('bndbox')
        # coords start at 1 in annotations, 0 in keras/tf
        xmin = int(float(bndbox.find('xmin').text)) - 1
        xmax = int(float(bndbox.find('xmax').text)) - 1
        ymin = int(float(bndbox.find('ymin').text)) - 1
        ymax = int(float(bndbox.find('ymax').text)) - 1
        difficult = int(object.find('difficult').text) == 1
        box = Box(xmin, ymin, xmax, ymax)
        gt_box = GroundTruthBox(obj_cls=name, difficult=difficult, box=box)
        gt_boxes.append(gt_box)

    img_metadata = Metadata(img_num, width=width, height=height, gt_boxes=gt_boxes, image_path=image_path)

    return img_metadata


def extract_img_data(base_path, img_num):
    metadata = extract_img_metadata(base_path, img_num)

    image = Image(metadata=metadata)
    return image


def get_img_names_from_set(base_path, set_name):
    img_set_base_path = os.path.join(base_path, IMAGESETS_DIR)
    img_set_path = os.path.join(img_set_base_path, set_name + '.txt')
    with open(img_set_path) as img_set_file:
        img_names = [line.rstrip('\n') for line in img_set_file]

    return img_names
