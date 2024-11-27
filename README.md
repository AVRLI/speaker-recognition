This repository contains two projects:

1. VAD (Tensorflow)
2. Speaker Verification on the STM32L4R9I-DISCOVERY Board (PyTorch)

The Speaker Verification project includes C and Python folders.
To use the model on the evaluation board, follow these steps:

1. Burn the C folder onto the board using the ST IDE.
2. Load the python/onnx/model_onnx into the X-CUBE-AI package.
3. Extract the Middlewares and X-CUBE-AI folders from the software.
4. Debug the code.

A demo is provided here.
After the BIP sound, we can hear the enrolled speaker's audio.
In the upper right corner of the screen, we can see the cosine similarity score between the enrolled embeddings and the embedding generated every second.
If the score exceeds the threshold, a LED lights up:

Orange LED: Indicates the enrolled male speaker.
Green LED: Indicates the enrolled female speaker.
