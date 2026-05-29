import torch
import cv2
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def generate_gradcam(model, input_tensor, rgb_img, target_layer):
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor)[0]

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return visualization