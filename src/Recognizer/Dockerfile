### Uncomment for Node.js functions
# FROM public.ecr.aws/lambda/nodejs
# ARG FUNCTION_DIR="/var/task"
# COPY index.js package.json ./
# RUN npm install
# CMD ["index.handler"]

### Uncomment for Python functions
# FROM public.ecr.aws/lambda/python
# COPY handler.py requirements.txt ./
# RUN python3 -m pip install -r requirements.txt
# CMD ["handler.handler"]

### Uncomment for Java functions
# FROM public.ecr.aws/lambda/java
# ARG SCRATCH_DIR=/var/task/build
# COPY src/ src/
# COPY gradle/ gradle/
# COPY build.gradle gradlew ./
# RUN mkdir build
# COPY gradle/lambda-build-init.gradle ./build
# RUN ./gradlew --project-cache-dir $SCRATCH_DIR/gradle-cache -Dsoftware.amazon.aws.lambdabuilders.scratch-dir=$SCRATCH_DIR --init-script $SCRATCH_DIR/lambda-build-init.gradle build
# RUN rm -r $SCRATCH_DIR/gradle-cache
# RUN rm -r $SCRATCH_DIR/lambda-build-init.gradle
# RUN cp -r $SCRATCH_DIR/*/build/distributions/lambda-build/* .
# FROM 628053151772.dkr.ecr.sa-east-1.amazonaws.com/awslambda/java11-runtime:beta
# COPY --from=build-image /var/task/META-INF ./
# COPY --from=build-image /var/task/handler ./handler
# CMD ["Handler::handler"]

### Uncomment for Go functions
# FROM golang:1.14 as build-image
# WORKDIR /go/src
# COPY go.mod main.go ./
# RUN go build -o ../bin
# FROM public.ecr.aws/lambda/go
# COPY --from=build-image /go/bin /var/task
# CMD ["function"]

### Uncomment for .NET functions
# FROM public.ecr.aws/lambda/dotnet
# COPY

# FROM mcr.microsoft.com/dotnet/core/sdk:3.1 as build-image
# ARG FUNCTION_DIR="/build"
# ENV PATH="/root/.dotnet/tools:${PATH}"
# RUN apt-get update && apt-get -y install zip
# RUN mkdir $FUNCTION_DIR
# WORKDIR $FUNCTION_DIR
# COPY Function.cs HelloWorld.csproj aws-lambda-tools-defaults.json $FUNCTION_DIR/
# RUN dotnet tool install -g Amazon.Lambda.Tools
# RUN dotnet lambda package
# FROM public.ecr.aws/lambda/dotnet
# COPY --from=build-image /build/bin/Release/netcoreapp3.1/publish/ /var/task/
# CMD ["HelloWorld::HelloWorld.Function::FunctionHandler"]

### Uncomment for Ruby functions
# FROM public.ecr.aws/lambda/ruby
# COPY function.rb Gemfile ./
# RUN bundle install
# CMD ["function.handler"]

### Uncomment for custom runtime functions
# FROM public.ecr.aws/lambda/provided
# COPY <sources> ./
# RUN <build command>
# ENTRYPOINT <executable>
