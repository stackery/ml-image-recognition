# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright Stackery, Inc. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import base64
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont
import requests
import tensorflow as tf

# Detect and annotate top N objects
NUM_OBJECTS = 3

# Loading model
loaded_model = tf.saved_model.load('model')
detector = loaded_model.signatures['default']

# Loading font
font = ImageFont.truetype('font/OpenSans-Regular.ttf', 25)

def lambda_handler(event, context):
    # Get image URL from `url` querystring parameter
    r = requests.get(event['queryStringParameters']['url'])

    # Detect objects from image
    objects = detect_objects(r.content)

    # Annotate objects onto image
    img = annotate_image(r.content, objects)

    # Encode image back into original format
    with BytesIO() as output:
        img.save(output, format=img.format)
        body = output.getvalue()

    # Send 200 response with annotated image back to client
    return {
        'statusCode': 200,
        'isBase64Encoded': True,
        'headers': {
            'Content-Type': img.get_format_mimetype()
        },
        'body': base64.b64encode(body)
    }

def detect_objects(image_content):
    img = tf.io.decode_image(image_content)

    # Executing inference.
    converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    result = detector(converted_img)
    
    return [
        {
            # TF results are in [ ymin, xmin, ymax, xmax ] format, switch to [ ( xmin, ymin ), ( xmax, ymax ) ] for PIL
            'box': [ ( result['detection_boxes'][i][1], result['detection_boxes'][i][0] ), ( result['detection_boxes'][i][3], result['detection_boxes'][i][2] ) ],
            'score': result['detection_scores'][i].numpy(),
            'class': result['detection_class_entities'][i].numpy().decode('UTF-8')
        }  for i in range(NUM_OBJECTS)
    ]

def annotate_image(image_content, objects):
    img = Image.open(BytesIO(image_content))
    draw = ImageDraw.Draw(img)

    for object in objects:
        # Multiply input coordinates, which range from 0 to 1, to number of pixels
        box = [ ( object['box'][0][0] * img.width, object['box'][0][1] * img.height ), ( object['box'][1][0] * img.width, object['box'][1][1] * img.height ) ]

        # Draw red rectangle around object
        draw.rectangle(box, outline='red', width=5)

        # Create label text and figure out how much space it uses
        label = f"{object['class']} ({round(object['score'] * 100)}%)"
        label_size = font.getsize(label)

        # Draw background rectangle for label
        draw.rectangle([ box[0], ( box[0][0] + label_size[0], box[0][1] + label_size[1]) ], fill='red')

        # Draw label text
        draw.text(box[0], label, fill='white', font=font)

    return img;