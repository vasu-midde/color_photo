# Black-and-White Photo Colorization

Adds colour to grayscale photos with a pre-trained deep CNN, run through OpenCV's DNN module.

The model is the one from **"Colorful Image Colorization"** by Zhang, Isola and Efros (ECCV 2016). This repo wires that pre-trained Caffe model into a small command-line tool. I did not train the model.

## How it works

1. Convert the input image to **Lab** colour space and keep the lightness channel **L**.
2. Resize L to 224×224 and feed it to the network.
3. The network predicts the **a/b** colour channels, using 313 quantised colour bins (`pts_in_hull.npy`).
4. Upscale a/b to the original size, merge with L, and convert back to BGR.

## Setup

```bash
pip install numpy opencv-python
```

Download the model weights (about 125 MB, not stored in this repo) into `model/`:

```
model/colorization_release_v2.caffemodel
```

The download link is in [`model/readme.md`](model/readme.md).

Note: `colorize.py` sets `PROJECT_DIR` to an absolute Windows path. Change it to the folder you cloned into.

## Run

```bash
python colorize.py -i ./images/bus.jpg
```

Sample inputs are in `images/`: `bus.jpg`, `rock.jpg`, `sky.jpeg`.

## Tech

Python · OpenCV (DNN) · NumPy · pre-trained Caffe CNN

## Credit

Model and method: R. Zhang, P. Isola, A. A. Efros, *Colorful Image Colorization*, ECCV 2016. [Project page](https://richzhang.github.io/colorization/)
