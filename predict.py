import pandas as pd
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import bone, figure
from PIL import Image
import difflib
import re
import math
import json
import sys
import argparse

import torch

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg

from PaddleOCR import PaddleOCR, draw_ocr
import heapq

# from VietnameseOcrCorrection.tool.predictor import Corrector
# import time
# from VietnameseOcrCorrection.tool.utils import extract_phrases

# from ultis import display_image_in_actual_size


# Specifying output path and font path.
FONT = './PaddleOCR/doc/fonts/latin.ttf'

import google.generativeai as genai

# Initialize Gemini model (only once)
genai.configure(api_key="AIzaSyDg-QwX8nhAMYF71jZhD1NgMNOnxTFxvDY")
gemini_model = genai.GenerativeModel("models/gemini-2.0-flash")


def predict(recognitor, detector, img_path, padding=4, refine_count=5):
    img = cv2.imread(img_path)

    # OCR detection
    result = detector.ocr(img_path, cls=False, det=True, rec=False)
    result = result[:][:][0]

    boxes = []
    for line in result:
        boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
    boxes = boxes[::-1]

    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        centre_x = (x1 + x2) / 2
        centre_y = (y1 + y2) / 2
        len_x = (x2 - x1) * 1.25
        len_y = (y2 - y1) * 1.25
        box[0][0] = int(centre_x - len_x / 2) - padding
        box[0][1] = int(centre_y - len_y / 2) - padding
        box[1][0] = int(centre_x + len_x / 2) + padding
        box[1][1] = int(centre_y + len_y / 2) + padding

    # Step 1: Recognize text and collect results
    results = []
    for box in boxes:
        x1, y1 = box[0]
        x2, y2 = box[1]
        src_pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        width, height = x2 - x1, y2 - y1
        dst_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))

        try:
            cropped_image = Image.fromarray(warped)
        except:
            continue

        text, prob = recognitor.predict(cropped_image, return_prob=True)
        results.append((box, text, prob, cropped_image))

    lowest_conf = heapq.nsmallest(refine_count, results, key=lambda x: x[2])

    refined_map = {}
    for box, orig_text, prob, crop in lowest_conf:
        try:
            gemini_response = gemini_model.generate_content([
                "Please read and transcribe the text in this image accurately. Only return the text, no other information.",
                crop
            ])
            refined_text = gemini_response.text.strip()
            refined_map[tuple(map(tuple, box))] = refined_text
        except Exception as e:
            print("Gemini error:", e)
            refined_map[tuple(map(tuple, box))] = orig_text  # fallback

    # Step 4: Reassemble final text results
    final_texts = []
    for box, text, prob, crop in results:
        key = tuple(map(tuple, box))
        final_texts.append(refined_map.get(key, text))

    for text in final_texts:
        print(text)
    return boxes, final_texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--output', required='./runs/predict', help='path to save output file')
    parser.add_argument('--use_gpu', required=False, help='is use GPU?')
    args = parser.parse_args()

    # Configure of VietOCR
    # Default weight
    config = Cfg.load_config_from_name('vgg_transformer')
    # Custom weight
    # config = Cfg.load_config_from_file('vi00_vi01_transformer.yml')
    # config['weights'] = './pretrain_ocr/vi00_vi01_transformer.pth'

    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    config['device'] = 'cuda'

    recognitor = Predictor(config)

    # Config of PaddleOCR
    detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=True)
    

    # Predict
    bounding_boxes, result_texts = predict(recognitor, detector, args.img, padding=0)


if __name__ == "__main__":    
    main()
