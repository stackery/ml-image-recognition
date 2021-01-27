# [Stackery](https://stackery.io) Machine Learning Image Recognition Example

This is an example machine learning image recognition stack using Lambda Container Images. Container images can include more source assets than traditional ZIP packages (10 GB vs 250 MB image sizes), allowing for larger ML models to be used.

This example contains an AWS Lambda function that uses the [Open Images Dataset](https://storage.googleapis.com/openimages/web/index.html) [TensorFlow](https://www.tensorflow.org/) model to detect objects in an image. When you invoke the function with a URL to an image the function will download the image and use the tensorflow model to detect objects in it.

Here is an overview of the files in this repo:

```text
.
├── .gitignore                         <-- Gitignore for Stackery
├── .stackery-config.yaml              <-- Default CLI parameters for root directory
├── LICENSE                            <-- MIT!
├── README.md                          <-- This README file
├── src
│  └── Recognizer
│     ├── Dockerfile                   <-- Dockerfile for building Recognizer Function
│     ├── handler.py                   <-- Recognizer Function Python source
│     └── requirements.txt             <-- Recognizer Function Python dependencies
└── template.yaml                      <-- SAM infrastructure-as-code template
```


## HOW TO
The repository contains a complete example stack that can be imported directly into Stackery and deployed. The rest of this README walks you through building your own ML stack using Stackery.

### Setup
1. First, create an AWS account at https://portal.aws.amazon.com/billing/signup if you don't have one. Don't worry, everything we will do in this example will fit into the [AWS Free Tier](https://aws.amazon.com/free), and even if you do not qualify for the free tier you will incur fractions of a penny's worth of expense running this example stack.
1. Second, create a free account at https://stackery.io/sign-up. We will use Stackery to build the app and then deploy it into your AWS account.
1. Follow the steps after creating your Stackery account to link your AWS account. This gives Stackery permissions to package functions and generate [CloudFormation](https://aws.amazon.com/cloudformation/) [Change Sets](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-updating-stacks-changesets.html) but does not give Stackery permission to arbitrarily modify resources in your AWS account. You can find more details here: [Stackery Permissions & Controls](https://docs.stackery.io/docs/using-stackery/security-permissions-and-controls).

### Create a new stack
1. Once you have linked your AWS account it's time to start creating a new stack! Go to https://app.stackery.io/stacks and click the green **Add a Stack** button. This will create a new, untitled stack for you.
1. First things first, let's give our stack a name. Click the three dots next to the current stack's name (the current name will be something like *Untitled* or *Untitled-1*) and choose to **Rename** the stack. Now, pick a name (*ml-image-recognition* would work!) and save it.
1. Before we go any further, why don't we save our progress? Click the big green **Save with Git...** button to authenticate to your favorite Git provider, create a new repo for the stack, and commit our starting stack.

### Add a new function
1. We see a blank canvas for our new stack. Let's add a Lambda Function! Click the green **Add Resource** button in the top right corner of the canvas. Click the **Function** resource in the palette to add a new Function to our stack. Close the palette.
1. Double-click the new Function to edit its settings.
1. Update the following settings:
    * Logical ID: Something descriptive, like *Recognizer*
    * Package Type: Docker Image (so we can fit a large ML model into our source assets!)
    * DockerFile: You probably want to update this from */src/Function/Dockerfile* to something like */src/Recognizer/Dockerfile*
    * Docker Build Context: You probably also want to update this from */src/Function* to something like */src/Recognizer*
    * Timeout: 120 (it takes a while for TensorFlow to spin up!)
    * Don't forget to click the big, green **Save** button at the bottom of the settings panel!
1. Save the stack by clicking the **Commit...** button. This is a great opportunity to see what we've added to the [AWS SAM](https://aws.amazon.com/serverless/sam/) template by opening the Template Diff panel. Notice how a bunch of additional resources, like an [AWS ECR](https://aws.amazon.com/ecr/) container image repository and a [CodeBuild](https://aws.amazon.com/codebuild/) project, are scaffolded to support building and running your new Function?

### Edit your sources
1. Now it's time to get our feet wet on the source code side! Pull down the stack's Git repo to edit it using your favorite editor / IDE.
1. Scaffolded in the repo will be your function's source folder and Dockerfile. This Dockerfile will have a lot of possible starting points for you to choose from. We're going to build a Python Function, so you can take a look at the default Python Dockerfile commands. But since we're building a specific implementation, delete what's in the Dockerfile and replace it with the following:
    ```Dockerfile
    # Download the Open Images TensorFlow model and extract it to the `model`
    # folder.
    FROM alpine AS builder
    RUN mkdir model
    RUN wget -c https://storage.googleapis.com/tfhub-modules/google/openimages_v4/ssd/mobilenet_v2/1.tar.gz -O - | tar xz -C model
    # Make sure it's world-readable so the Lambda service user can access it.
    RUN chmod -R a+r model

    # Build the runtime image from the official AWS Lambda Python base image.
    FROM public.ecr.aws/lambda/python
    # Copy the extracted Open Images model into the source code space.
    COPY --from=builder model model
    # Copy in the sources.
    COPY handler.py requirements.txt ./
    # Install the Python dependencies
    RUN python3 -m pip install -r requirements.txt
    # Tell the Lambda runtime where the function handler is located.
    CMD ["handler.lambda_handler"]
    ```
1. Now add the Python dependencies by putting the following into requirements.txt:
    ```
    requests
    tensorflow
    ```
1. Lastly, add the Function handler source code into handler.py:
    ```python
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
    ```
1. Save, add, commit, and push these changes up to your stack's repo.

### Deploy the stack
1. Now it's time to deploy the stack! If you have the `stackery` CLI (you can install it from instructions here: https://docs.stackery.io/docs/using-stackery/cli#install-the-cli) it's as easy as running `stackery deploy --git-ref main` (swap `main` for whatever branch your code is on). You can also point and click through our dashboard interface to deploy by navigating to the **Deploy** section on the left-hand sidebar after navigating to your stack at https://app.stackery.io.

### Test
You can easily test two ways: using the AWS CLI or the AWS Lambda console.

#### Using the AWS CLI
1. Find the name of your function after you deploy. If you used the Stackery CLI to deploy it will be printed out for you at the end of the deployment process. You can also find it in the Stackery dashboard by navigating to the **View** section on the left-hand sidebar after navigating to your stack at https://app.stackery.io. Double-click on your Function in the canvas view to find it's name.
1. Run:
    ```
    aws lambda invoke --function-name <your function name here> --cli-binary-format raw-in-base64-out --payload '{"url":"https://images.pexels.com/photos/310983/pexels-photo-310983.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940"}' /tmp/response.json
    ```
1. After some time (up to a minute and a half) you will see the response from the Lambda service. Hopefully it will show success, with a 200 Status Code like we see in the following output:
    ```json
    {
        "StatusCode": 200,
        "ExecutedVersion": "$LATEST"
    }
    ```
    * If not, you can run `stackery logs` to show the logs for your function and figure out what may be wrong.
1. To see the resulting set of objects found in the image, view the /tmp/response.json file.

#### Using the AWS Console
1. Find the deployed Function in the Stackery dashboard by navigating to the **View** section on the left-hand sidebar after navigating to your stack at https://app.stackery.io. Double-click on your Function in the canvas view to open its properties. Click the green **View in AWS Console** button to open the AWS Lambda Console.
1. Click the **Test** button in the top right corner of the AWS console. It will ask you to create a new test event. Create a name for the event (*test* would be fine enough!) and put the following in the test body:
    ```json
    {
      "url": "https://images.pexels.com/photos/310983/pexels-photo-310983.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940"
    }
    ```
1. Save the new test event.
1. Click the same **Test** button in the AWS console again. Notice the new **Execution result** dialog that was created at the top of the console. It will remain disabled while the function executes, which can take up to two minutes. Once it completes you can expand the results dialog to view the detected object data and execution metrics and logs.

### Give a shout out!
We love hearing if we've helped folks learn more about AWS, serverless, or any other aspect of building this ML project! If you're so inclined, give us a shout out on Twitter [@stackeryio](https://twitter.com/stackeryio). We'd love to send some thanks your way, too!