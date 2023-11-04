# Image-Captioning

This repository contains code for an image captioning model trained on the Flickr8k dataset. The model generates textual descriptions for images.

## Dataset

The Flickr8k dataset contains 8000 images along with 5 captions per image. The images are divided into train, validation, and test splits.

## Models

Two models are implemented:

- Baseline - Encoder-decoder architecture with CNN encoder (ResNet-50) and RNN decoder (LSTM)
- Attention - Encoder-decoder with attention mechanism. Encoder is ResNet-101 and decoder is LSTM with spatial attention.

## Usage

The Jupyter notebooks contain the implementation:

- ```baseline.ipynb``` : Notebook to train and evaluate baseline model
- ```attention.ipynb``` : Notebook to train and evaluate attention model

To train a model:

- Run all cells in the notebook
- Model checkpoints will be saved
- Use saved checkpoints for evaluation

To evaluate a pretrained model:
- Load saved model checkpoint
- Run evaluation cell

## Evaluation

The models are evaluated using BLEU and METEOR metrics.

On the test set, the attention model achieves higher BLEU and METEOR scores compared to the baseline.

## References
Flickr8k dataset paper:

Hodosh, Micah, Peter Young, and Julia Hockenmaier. "Framing image description as a ranking task: Data, models and evaluation metrics." Journal of Artificial Intelligence Research 47 (2013): 853-899
