# Advanced Multimodal Transformer Model

This project contains an advanced multimodal transformer capable of processing real video, audio, and text data. The model fuses the provided modalities to perform classification.

## Features

- Process real video files
- Process real audio files
- Parse accompanying text descriptions
- Fuse all modalities for classification
- Use an enhanced 3D-CNN backbone for video
- Apply advanced spectrogram processing for audio
- Provide BERT-based text support

## Project Layout

- `basic-multimodal.py`: Main application script
- `requirements.txt`: Required Python packages
- `multimodal_dataset/`: Dataset folder
  - `videos/`: Video files
  - `audios/`: Audio files
  - `texts/`: Text files
  - `metadata.json`: Dataset metadata

## Installation

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

The model can run with either sample data or your own video, audio, and text files:

```bash
python basic-multimodal.py
```

When the program starts it offers two options:
1. Sample data (automatically generated demo)
2. Real data (provide your own video, audio, and text files)

If you choose real data:
1. Place video files in `multimodal_dataset/videos/`
2. Place audio files in `multimodal_dataset/audios/`
3. Provide text descriptions for each example or add files to `multimodal_dataset/texts/`

## Model Architecture

The multimodal model consists of three primary components:

1. **Video Encoder**: Extracts features from videos using a 3D CNN
   - 224×224 resolution
   - 16 frames per clip
   - Adaptive average pooling and dropout layers

2. **Audio Encoder**: Extracts features from spectrograms using a 2D CNN
   - Mel spectrogram inputs
   - 128 mel filter bands
   - 5-second audio segments

3. **Text Encoder**: Uses a BERT model to produce text embeddings

For multimodal fusion the pipeline includes:
- Transformer-based cross-attention
- Multi-head attention
- Layer normalisation

## Outputs

Training artefacts, visualisations, and the trained model are saved inside the `multimodal_dataset/` directory.

## Requirements

- Python 3.7+
- PyTorch 1.9+
- transformers
- torchaudio
- torchvision
- OpenCV
- NumPy
- Matplotlib
