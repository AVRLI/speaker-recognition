This repository contains two projects:

1. VAD from Scratch
2. Speaker Verification on the STM32L4R9I-DISCOVERY Board

The Speaker Verification project includes C and Python folders.
To use the model on the evaluation board, follow these steps:

1. Burn the C folder onto the board using the ST IDE.
2. Load the python/onnx/onnx_model into the X-CUBE-AI package.
3. Extract the Middlewares and X-CUBE-AI folders from the software.
4. Debug the code.

A demo is provided here.
After the BIP sound, we can hear the enrolled speaker's audio.

In the upper left corner of the screen, you can see the cosine similarity score between the enrolled embeddings and the embedding generated every second.
If the score exceeds the threshold, an LED lights up:

Orange LED: Indicates the enrolled male speaker.
Green LED: Indicates the enrolled female speaker.
