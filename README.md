
This repository contains two projects:

Voice Activity Detection (VAD) - Implemented in TensorFlow.

Speaker Verification - implemented in PyTorch and deployed on the STM32L4R9I-DISCOVERY Board.

The Speaker Verification project includes:

A C folder containing code for deployment on the STM32L4R9I-DISCOVERY board.
A Python folder containing Jupyter notebooks for training the model.

To train the speaker verification model, follow the Jupyter notebooks provided in the python folder. The model is trained on the VoxCeleb1 training dataset, validated on the VoxCeleb1 validation dataset, and can be tested on the VoxCeleb2 test dataset. Verification lists are available in the dataset list folder. Note that the VoxCeleb2 data may need to be converted to WAV format using ffmpeg before testing. The data can be downloaded from: https://mm.kaist.ac.kr/datasets/voxceleb/

To deploy and use the model on the STM32L4R9I-DISCOVERY evaluation board, follow these steps:

Flash the contents of the C folder onto the STM32 board using the ST IDE software.

Use the X-CUBE-AI package to load the ONNX model file located at python/onnx/model_onnx.

Extract the Middlewares and X-CUBE-AI folders from the software.

Use the IDE to debug and run the code on the board.

A demo is provided here:
https://youtu.be/Nu4O4an-qco
After the BIP sound, we can hear the enrolled speaker's voice.
In the upper right corner of the screen, we can see the cosine similarity score between the enrolled embeddings and the embedding generated every second.
If the score exceeds the threshold, a LED lights up:
Orange LED: Indicates the enrolled male speaker.
Green LED: Indicates the enrolled female speaker.
