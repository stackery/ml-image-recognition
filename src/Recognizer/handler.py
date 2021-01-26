# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright Stackery, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import requests
import tensorflow as tf

# Loading model
loaded_model = tf.saved_model.load('model')
detector = loaded_model.signatures['default']

def lambda_handler(event, context):
    r = requests.get(event['url'])
    img = tf.image.decode_jpeg(r.content, channels=3)

    # Executing inference.
    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)

    return {
        'detection_boxes' : result['detection_boxes'].numpy().tolist(),
        'detection_scores': result['detection_scores'].numpy().tolist(),
        'detection_class_entities': [el.decode('UTF-8') for el in result['detection_class_entities'].numpy()] 
    }