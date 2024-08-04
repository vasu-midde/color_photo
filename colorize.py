import numpy as np
import argparse
import cv2
import os

# Define the paths to the model files
PROJECT_DIR = r"C:/Users/chintu/OneDrive/Desktop/color_photo"
PROTO_FILE = os.path.join(PROJECT_DIR, "model/colorization_deploy_v2.prototxt")
POINTS_FILE = os.path.join(PROJECT_DIR, "model/pts_in_hull.npy")
CAFFE_MODEL = os.path.join(PROJECT_DIR, "model/colorization_release_v2.caffemodel")

# Set up argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True,
	help="Path to the input grayscale image")
arguments = vars(parser.parse_args())

# Validate the existence of model files
if not os.path.isfile(PROTO_FILE):
    raise FileNotFoundError(f"Prototxt file not found at {PROTO_FILE}")
if not os.path.isfile(POINTS_FILE):
    raise FileNotFoundError(f"Points file not found at {POINTS_FILE}")
if not os.path.isfile(CAFFE_MODEL):
    raise FileNotFoundError(f"Model file not found at {CAFFE_MODEL}")

# Load the Caffe model
print("Loading the model...")
network = cv2.dnn.readNetFromCaffe(PROTO_FILE, CAFFE_MODEL)
points = np.load(POINTS_FILE)

# Set up the network with quantized centers for ab channels
layer_class8 = network.getLayerId("class8_ab")
layer_conv8 = network.getLayerId("conv8_313_rh")
points = points.transpose().reshape(2, 313, 1, 1)
network.getLayer(layer_class8).blobs = [points.astype("float32")]
network.getLayer(layer_conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Load and preprocess the input image
input_image = cv2.imread(arguments["image"])
if input_image is None:
    raise FileNotFoundError(f"Input image not found at {arguments['image']}")

image_scaled = input_image.astype("float32") / 255.0
lab_image = cv2.cvtColor(image_scaled, cv2.COLOR_BGR2LAB)

lab_resized = cv2.resize(lab_image, (224, 224))
L_channel = cv2.split(lab_resized)[0]
L_channel -= 50

# Perform colorization
print("Colorizing the image...")
network.setInput(cv2.dnn.blobFromImage(L_channel))
ab_channels = network.forward()[0, :, :, :].transpose((1, 2, 0))

ab_resized = cv2.resize(ab_channels, (input_image.shape[1], input_image.shape[0]))

L_original = cv2.split(lab_image)[0]
colorized_image = np.concatenate((L_original[:, :, np.newaxis], ab_resized), axis=2)

colorized_bgr = cv2.cvtColor(colorized_image, cv2.COLOR_LAB2BGR)
colorized_bgr = np.clip(colorized_bgr, 0, 1)

final_image = (255 * colorized_bgr).astype("uint8")

# Display the original and colorized images
cv2.imshow("Original Image", input_image)
cv2.imshow("Colorized Image", final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
