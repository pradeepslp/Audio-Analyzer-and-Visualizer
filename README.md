# Audio Analyzer and Visualizer

## Project Overview
The Audio Analyzer and Visualizer is a Python-based application that analyzes audio files and displays their audio characteristics visually. Users can upload audio files, listen to them, view waveforms, and analyze their frequency spectrum.

## Features
- Upload WAV and MP3 audio files.
- Play uploaded audio.
- Display audio duration, sample rate, and number of channels.
- Visualize the audio waveform.
- Display the frequency spectrum using Fast Fourier Transform (FFT).
- Handle invalid files and errors.

## Technologies Used
- Python
- Streamlit
- Librosa
- NumPy
- Matplotlib

## Project Structure
```text
Audio-Analyzer-and-Visualizer/
├── app.py
├── requirements.txt
└── README.md
```

## Installation

**Step 1: Install Python**

Install Python 3.10 or later and ensure Python is added to PATH.

**Step 2: Install dependencies**

Open the terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

**Step 3: Run the application**

```bash
streamlit run app.py
```

The application will open in your web browser.

## How to Use
1. Open the application.
2. Upload a WAV or MP3 audio file.
3. Play the audio using the built-in audio player.
4. View the audio duration, sample rate, and channel information.
5. Examine the waveform visualization.
6. View the frequency spectrum to understand the audio's frequency components.

## Applications
- Audio signal analysis
- Speech analysis
- Educational demonstrations
- Basic digital signal processing
- Audio waveform and frequency visualization

## Conclusion
The Audio Analyzer and Visualizer provides a simple and interactive way to explore audio signals using Python. It demonstrates fundamental concepts of audio processing, waveform visualization, and frequency analysis.

## Author
Student Project – Text and Speech Analysis (Zero Gap)
