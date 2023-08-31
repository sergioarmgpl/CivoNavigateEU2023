
FROM debian:bullseye
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update;apt-get install apt-utils -y;apt-get install curl gnupg gnupg2 gnupg1 -y
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install libedgetpu1-std -y
RUN apt-get install python3-pycoral python3-pip -y
RUN apt-get install git -y
RUN git clone https://github.com/google-coral/pycoral.git
#RUN usermod -aG plugdev $USER
WORKDIR pycoral
COPY . .
COPY owl.jpg test_data
COPY quetzal.jpg test_data
COPY eagle.jpg test_data
RUN pip3 install -r requirements.txt
RUN bash examples/install_requirements.sh classify_image.py
EXPOSE 5000
CMD python3 index.py
#CMD python3 examples/classify_image.py --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --labels test_data/inat_bird_labels.txt --input test_data/parrot.jpg
