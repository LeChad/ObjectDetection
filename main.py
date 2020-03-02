import configparser
import cv2
import torch
from torch.autograd import Variable
from additional.darknet import Darknet
import additional.util as utility
import random

def Main():
    """ Main function for object detection project"""

    # 1.) Check to see if this is a CUDA-capable system
    if torch.cuda.is_available():
        # 2.) Load Settings in settings.cfg
        config = configparser.ConfigParser()
        config.read('configuration/settings.cfg')

        weights_file = config['main']['weights_file']
        nms = float(config['main']['nms']) # Non Max Suppression. Gets rid of duplicate detections (overlapping).
        yolo_config = config['main']['yolo_config']
        confidence = float(config['main']['confidence'])  # Confidence (IOU) prediction, decimal format. 70% = 0.70.
        resolution = int(config['main']['resolution'])  # View documentation for information *resolution is important*.

        # 3.) Load Pretrained Neural Network based on COCO Datasets
        print("Loading the network....")
        model = Darknet(yolo_config)  # Load yolov3.cfg
        model.load_weights(weights_file)  # Load yolov3.weights

        global classes  # Create global classes variable so we can call it when writing to cv2 screen.
        classes = utility.load_classes('additional/coco.names')  # Load coco name classification file

        print("Assigning classified objects a random RGB color.")
        global classes_dictionary # creates global classes dictionary.
        classes_dictionary = dict()  # create a dictionary for object classification and rgb color assignment based on random intergers

        for object_class in classes:
            if object_class not in classes_dictionary:
                classes_dictionary[object_class] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # Generate random rgb for object

        model.cuda()  # move model to GPU
        model.eval()  # change to eval mode.

        print("Neural network has been successfully loaded!")
        # If errors, implement try-exception to catch it and produce exact error.

        resolution_verification = int(resolution / 13)  # YOLO algorithm bounding boxes function off of 13x13. Verify resolution.
        if (resolution % resolution_verification) == 0:
            model.net_info['height'] = resolution  # Set resolution.
            model.net_info['widght'] = resolution

            capture = cv2.VideoCapture(0)  # capture stream from webcam (might need to change 0 to whatever index your camera is)
            while capture.isOpened():
                ret, frame = capture.read()

                if ret:
                    image, original_image, image_dimensions = ProcessImage(frame, resolution)  # Process image with resolution settings.

                    # Code below is passing the image through Darknet & Yolo for processing.
                    image = image.cuda() # Convert the image to Tensor format for processing.
                    detections = model(Variable(image), torch.cuda.is_available())  # Get object detection predictions

                    detections = utility.write_results(detections, confidence, len(classes_dictionary), nms=True, nms_conf=nms)  # obtain predictions based off of confidence and nms settings, 80 is number of classes to use (COCO)

                    # Set bounding boxes around detected objects
                    detections[:, 1:5] = torch.clamp(detections[:, 1:5], 0.0, float(resolution)) / resolution  # Restricts tensors to specified range, return Tensor format.
                    detections[:, [2, 4]] *= frame.shape[0]  # height of detected object
                    detections[:, [1, 3]] *= frame.shape[1]  # width of detected object

                    list(map(lambda x: WriteToCv2(x, original_image), detections))  # iters through detections, applies boxes and labels

                    cv2.imshow("Object Detection Stream", original_image)
                    key = cv2.waitKey(1)

                    if key & 0xFF == ord('q'):
                        break

        else:
            print("[ERROR] - Inappropriate resolution. Please view documentation.")
    else:
        print("CUDA is unavailable. Please visit Nvidia 'How-to-CUDA' for more information or see project documentation.")


def WriteToCv2(x, image):
    """ Draws boxes and labels around objects that have been detected. """
    point1 = tuple(x[1:3].int())  # Draw cv2 rectangle based on object Tensor Locations
    point2 = tuple(x[3:5].int())  # Draw Cv2 rectangle based on object Tensor Location

    object_class_index = int(x[-1])  # Object class index (human, chair, sofa, etc etc)

    label = "{0}".format(classes[object_class_index])  # what the label is eg: person, cat, dog, bus

    color = classes_dictionary[label] # Give the object a color based off of pre-established dictionary.
    cv2.rectangle(image, point1, point2, color, 2)  # draw the rectangle for object
        # rectangle(image, point1, point2, color of box, thickness)

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    point2 = (point1[0] + text_size[0] + 3), (point1[1] + text_size[1] + 4)

    cv2.rectangle(image, point1, point2, color, -1)  # Draw rectangle for label
    cv2.putText(image, label, (point1[0], point1[1] + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)  # Write text label to drawn rectangle.
        #putText(image, text, origin points (offset text), font, fontscale, color, thickness)
        # (point1[0], point1[1] + text_size[1] + 4) is the origin point for the text we are going to draw. Offset it by 4.

    return image

def ProcessImage(image_to_process, resolution_dimensions):
    """ Converts an image / frame to a Tensors """
    original_image = image_to_process  # Keep an unmodified copy of the frame
    image_dimensions = original_image.shape[1], original_image.shape[0]  # Width, height of capture frame.

    #image_to_process = cv2.resize(original_image, (resolution_dimensions, resolution_dimensions))  # Resize Image for Tensor processing.
    processed_image = torch.from_numpy(cv2.resize(original_image, (resolution_dimensions, resolution_dimensions)).transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
    return processed_image, original_image, image_dimensions


if __name__ == "__main__":
    Main()