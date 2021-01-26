FROM alpine AS builder
RUN mkdir model
RUN wget -c https://storage.googleapis.com/tfhub-modules/google/openimages_v4/ssd/mobilenet_v2/1.tar.gz -O - | tar xz -C model
RUN chmod -R a+r model

FROM public.ecr.aws/lambda/python
COPY --from=builder model model
COPY handler.py requirements.txt ./
RUN python3 -m pip install -r requirements.txt
CMD ["handler.lambda_handler"]