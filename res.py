"""
Integration of centerpose
"""

import time

import numpy as np

import cv2
import torch

from detector.centorpose.model import create_model, load_model


class ResDetector(BaseDetector):
  def __init__(self):
    pass

  def image_preprocess(self, img_name):
    pass

  def images_detection(self, imgs, orig_dim_list):
    pass

  def detect_one_img(self, img_name):
    pass

cfg = 




