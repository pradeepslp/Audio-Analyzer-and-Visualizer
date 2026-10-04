"""
Audio Analyzer and Visualizer
=============================
A modern, interactive Streamlit application to analyze and visualize audio files (WAV, MP3)
using Librosa, NumPy, and Matplotlib.

Features:
- Upload audio files in WAV or MP3 format
- Synthesize sample audio for instant demo testing
- Display audio duration, sample rate, number of channels, and audio metrics
- Time-domain waveform visualizer (Mono and Stereo support)
- Fast Fourier Transform (FFT) Frequency Spectrum Analyzer
- Short-Time Fourier Transform (STFT) Spectrogram view
- Built-in HTML5 audio player
- Clean, responsive glassmorphism UI with robust error handling
"""

import io
import time
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import soundfile as sf
import librosa

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Audio Analyzer & Visualizer",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern, Premium Dark-Theme Aesthetics
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, rgba(29, 38, 113, 0.45) 0%, rgba(195, 55, 100, 0.45) 100%);
        padding: 24px 30px;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(12px);
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.25);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38ef7d, #11998e, #00d2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .sub-title {
        font-size: 1.0rem;
        color: #b0bec5;
        margin-top: 6px;
        margin-bottom: 0;
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 16px 20px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 239, 125, 0.4);
        box-shadow: 0 6px 25px rgba(56, 239, 125, 0.1);
    }
    
    .metric-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #90a4ae;
        font-weight: 600;
        margin-bottom: 4px;
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.5px;
    }
    
    .metric-sub {
        font-size: 0.78rem;
        color: #78909c;
        margin-top: 2px;
    }

    /* Badges & Tags */
    .tag-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        background: rgba(0, 210, 255, 0.15);
        color: #00d2ff;
        border: 1px solid rgba(0, 210, 255, 0.3);
        margin-right: 6px;
    }

    /* Player Container */
    .player-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 18px 24px;
        margin-bottom: 25px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 6px 8px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        padding: 8px 18px;
    }

    /* Clean divider */
    hr {
        border-color: rgba(255, 255, 255, 0.08);
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Matplotlib Dark Theme Setup
# -----------------------------------------------------------------------------
def apply_plot_theme(fig, ax_list):
    """Apply sleek modern dark theme to matplotlib figures."""
    fig.patch.set_facecolor('#0E1117')
    for ax in ax_list:
        ax.set_facecolor('#151A22')
        ax.spines['bottom'].set_color('#30363D')
        ax.spines['top'].set_color('#30363D')
        ax.spines['left'].set_color('#30363D')
        ax.spines['right'].set_color('#30363D')
        ax.xaxis.label.set_color('#C9D1D9')
        ax.yaxis.label.set_color('#C9D1D9')
        ax.tick_params(axis='x', colors='#8B949E')
        ax.tick_params(axis='y', colors='#8B949E')
        ax.grid(True, linestyle='--', alpha=0.25, color='#8B949E')
        if ax.get_title():
            ax.title.set_color('#F0F6FC')


# -----------------------------------------------------------------------------
# Synthetic Demo Audio Generator (For zero-friction instant testing)
# -----------------------------------------------------------------------------
def generate_synthetic_audio(preset_type: str, duration_sec: float = 3.0, sr: int = 44100):
    """Generate high-quality synthetic audio for demonstration and testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    
    if preset_type == "Concert A (440 Hz Sine Wave)":
        # Pure sine wave with smooth attack/decay envelope
        y = 0.6 * np.sin(2 * np.pi * 440.0 * t)
        channels = 1
    elif preset_type == "C-Major Harmonic Chord (261.6 + 329.6 + 392.0 Hz)":
        # Multi-tone chord (C4, E4, G4)
        c4 = 0.3 * np.sin(2 * np.pi * 261.63 * t)
        e4 = 0.25 * np.sin(2 * np.pi * 329.63 * t)
        g4 = 0.25 * np.sin(2 * np.pi * 392.00 * t)
        c5 = 0.15 * np.sin(2 * np.pi * 523.25 * t)
        y = c4 + e4 + g4 + c5
        channels = 1
    elif preset_type == "Stereo Dual-Tone (440 Hz Left / 880 Hz Right)":
        # Stereo left and right differing frequencies
        left = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        right = 0.5 * np.sin(2 * np.pi * 880.0 * t)
        y = np.vstack([left, right])
        channels = 2
    elif preset_type == "Frequency Sweep / Chirp (100 Hz - 5000 Hz)":
        # Linear frequency chirp
        f0, f1 = 100.0, 5000.0
        phase = 2 * np.pi * (f0 * t + ((f1 - f0) / (2 * duration_sec)) * (t ** 2))
        y = 0.5 * np.sin(phase)
        channels = 1
    else:
        # Default simple tone
        y = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        channels = 1

    # Apply 20ms fade in / fade out to avoid clicks
    fade_len = int(sr * 0.02)
    if fade_len > 0:
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        if y.ndim == 1:
            y[:fade_len] *= fade_in
            y[-fade_len:] *= fade_out
        else:
            y[:, :fade_len] *= fade_in
            y[:, -fade_len:] *= fade_out

    # Encode to WAV BytesIO buffer for audio player and librosa loading
    out_buf = io.BytesIO()
    if y.ndim == 1:
        sf.write(out_buf, y, sr, format='WAV')
    else:
        sf.write(out_buf, y.T, sr, format='WAV')
    out_buf.seek(0)
    
    return out_buf, y, sr, channels


# -----------------------------------------------------------------------------
# Audio Loading & Analysis Helpers
# -----------------------------------------------------------------------------
def load_and_inspect_audio(file_bytes_or_buf):
    """
    Load audio data using Librosa & SoundFile.
    Returns:
        dict containing audio signal, sample rate, channels, duration, and metrics.
    """
    try:
        # Reset stream position if BytesIO
        if hasattr(file_bytes_or_buf, 'seek'):
            file_bytes_or_buf.seek(0)

        # Load audio with native sample rate and preserve stereo channels if present
        y, sr = librosa.load(file_bytes_or_buf, sr=None, mono=False)
        
        # Determine channel count and samples
        if y.ndim == 1:
            channels = 1
            n_samples = len(y)
            mono_signal = y
        elif y.ndim == 2:
            channels = y.shape[0]
            n_samples = y.shape[1]
            mono_signal = np.mean(y, axis=0) # Downmix to mono for universal FFT & STFT
        else:
            raise ValueError(f"Unsupported audio dimension: {y.ndim}")

        if n_samples == 0:
            raise ValueError("The audio file contains zero samples.")

        duration = n_samples / sr
        
        # Basic Metrics
        peak_val = np.max(np.abs(mono_signal))
        rms_val = np.sqrt(np.mean(mono_signal ** 2))
        dbfs_val = 20 * np.log10(peak_val + 1e-12)

        return {
            "success": True,
            "y": y,
            "mono_signal": mono_signal,
            "sr": sr,
            "channels": channels,
            "duration": duration,
            "n_samples": n_samples,
            "peak_val": peak_val,
            "rms_val": rms_val,
            "dbfs_val": dbfs_val,
            "error": None
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def compute_fft(signal: np.ndarray, sr: int):
    """
    Compute One-Sided Fast Fourier Transform (FFT) for real audio signal.
    Returns frequencies, linear magnitudes, and dB magnitudes.
    """
    n = len(signal)
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(n)
    windowed_signal = signal * window
    
    # Real FFT
    fft_vals = np.fft.rfft(windowed_signal)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    
    # Normalized Magnitude
    mag = np.abs(fft_vals) / (n / 2.0)
    mag[0] /= 2.0 # DC component normalization
    
    # Decibels relative to peak
    max_mag = np.max(mag) if np.max(mag) > 0 else 1.0
    mag_db = 20 * np.log10((mag / max_mag) + 1e-12)
    
    # Identify dominant peak frequency (ignoring DC near 0 Hz)
    search_idx = np.where(freqs >= 20.0)[0]
    if len(search_idx) > 0:
        sub_mag = mag[search_idx]
        peak_idx = search_idx[np.argmax(sub_mag)]
        peak_freq = freqs[peak_idx]
        peak_magnitude = mag[peak_idx]
    else:
        peak_freq = freqs[np.argmax(mag)]
        peak_magnitude = np.max(mag)
        
    return freqs, mag, mag_db, peak_freq, peak_magnitude


def calculate_frequency_bands(freqs: np.ndarray, mag: np.ndarray):
    """Calculate energy distribution across standard acoustic frequency bands."""
    bands = {
        "Sub-Bass (20-60 Hz)": (20, 60),
        "Bass (60-250 Hz)": (60, 250),
        "Low Mids (250-500 Hz)": (250, 500),
        "Midrange (500-2k Hz)": (500, 2000),
        "High Mids (2k-6k Hz)": (2000, 6000),
        "Treble (6k-20k Hz)": (6000, 20000)
    }
    
    band_energies = {}
    total_energy = np.sum(mag ** 2) + 1e-12
    
    for band_name, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if np.any(mask):
            energy = np.sum(mag[mask] ** 2)
            band_energies[band_name] = (energy / total_energy) * 100.0
        else:
            band_energies[band_name] = 0.0
            
    return band_energies


# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------
def main():
    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">Audio Analyzer & Visualizer</h1>
        <p class="sub-title">Explore waveforms, analyze frequency spectra using FFT, inspect audio channels, and visualize spectrograms in real time.</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sidebar: Source Selection & Settings
    # -------------------------------------------------------------------------
    st.sidebar.markdown("### ⚙️ Audio Input Source")
    
    input_mode = st.sidebar.radio(
        "Select Audio Source:",
        ["📁 Upload Audio File (WAV / MP3)", "🧪 Preset Synthetic Tone (Demo)"],
        index=0,
        help="Upload your own file or try an instantly generated audio waveform."
    )
    
    audio_buffer = None
    file_name = "audio"
    
    if input_mode == "📁 Upload Audio File (WAV / MP3)":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a WAV or MP3 file:",
            type=["wav", "mp3"],
            help="Supports standard WAV and MP3 audio files."
        )
        
        if uploaded_file is not None:
            # Check file size (e.g. limit to 50MB for responsive browser performance)
            if uploaded_file.size > 50 * 1024 * 1024:
                st.sidebar.error("⚠️ File size exceeds 50MB limit. Please upload a smaller audio clip.")
                return
            audio_buffer = uploaded_file
            file_name = uploaded_file.name
        else:
            st.info("👋 **Welcome!** Please upload an audio file (`.wav` or `.mp3`) from the sidebar, or select **Preset Synthetic Tone** to try sample audio right away.")
            
            # Show a helpful guide card
            st.markdown("""
            ### 📌 Quick Guide
            - **Upload audio** on the left sidebar to analyze duration, sample rate, channels, waveform, and FFT spectrum.
            - **Or choose a preset tone** to instantly see how sine waves, harmonic chords, stereo tones, and frequency sweeps appear in the frequency domain.
            - All processing is done locally on your machine with **NumPy**, **Librosa**, and **Matplotlib**.
            """)
            return

    else:
        # Synthetic Tone Generator
        preset_choice = st.sidebar.selectbox(
            "Select Demo Preset:",
            [
                "Concert A (440 Hz Sine Wave)",
                "C-Major Harmonic Chord (261.6 + 329.6 + 392.0 Hz)",
                "Stereo Dual-Tone (440 Hz Left / 880 Hz Right)",
                "Frequency Sweep / Chirp (100 Hz - 5000 Hz)"
            ]
        )
        demo_duration = st.sidebar.slider("Duration (seconds):", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
        
        with st.spinner("Synthesizing demo audio..."):
            synth_buf, synth_y, synth_sr, synth_ch = generate_synthetic_audio(preset_choice, duration_sec=demo_duration)
            audio_buffer = synth_buf
            file_name = f"{preset_choice.split()[0].lower()}_demo.wav"

    # Analysis Customization in Sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎛️ Visualization Settings")
    
    max_fft_freq = st.sidebar.slider(
        "FFT Max Display Frequency (Hz):",
        min_value=500,
        max_value=22050,
        value=5000,
        step=500,
        help="Zoom in on lower frequencies (e.g. speech / music fundamentals) or expand to full Nyquist limit."
    )
    
    fft_scale = st.sidebar.radio(
        "FFT Magnitude Scale:",
        ["Decibels (dBFS Normalized)", "Linear Amplitude"],
        index=0
    )

    spectrogram_cmap = st.sidebar.selectbox(
        "Spectrogram Color Map:",
        ["magma", "viridis", "plasma", "inferno", "cividis", "turbo"],
        index=0
    )

    # -------------------------------------------------------------------------
    # Audio Processing & Data Extraction
    # -------------------------------------------------------------------------
    with st.spinner("Analyzing audio data..."):
        analysis = load_and_inspect_audio(audio_buffer)

    if not analysis["success"]:
        st.error(f"❌ **Error reading audio file**: {analysis['error']}")
        st.warning("Please ensure the file is a valid, uncorrupted WAV or MP3 audio file.")
        return

    # Extract parsed data
    y = analysis["y"]
    mono_signal = analysis["mono_signal"]
    sr = analysis["sr"]
    channels = analysis["channels"]
    duration = analysis["duration"]
    n_samples = analysis["n_samples"]
    peak_val = analysis["peak_val"]
    rms_val = analysis["rms_val"]
    dbfs_val = analysis["dbfs_val"]

    # Compute FFT
    freqs, mag_linear, mag_db, peak_freq, peak_mag = compute_fft(mono_signal, sr)

    # -------------------------------------------------------------------------
    # Audio Player Section
    # -------------------------------------------------------------------------
    st.markdown("### 🎧 Audio Playback")
    
    # Make sure buffer pointer is at beginning for Streamlit audio playback
    if hasattr(audio_buffer, 'seek'):
        audio_buffer.seek(0)
    
    col_player, col_meta = st.columns([2, 1])
    with col_player:
        # Determine MIME type
        mime = "audio/wav" if file_name.lower().endswith(".wav") else "audio/mp3"
        st.audio(audio_buffer, format=mime)
        
    with col_meta:
        st.markdown(f"""
        <div style="padding-top: 8px;">
            <span class="tag-badge">📄 {file_name}</span>
            <span class="tag-badge">{'🔊 Stereo' if channels == 2 else '🔈 Mono'}</span>
            <span class="tag-badge">🎯 {sr:,} Hz</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr/>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Key Audio Properties & Metric Cards
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Audio Properties & Metrics")
    
    m_col1, m_col2, m_col3, m_col4, m_col5, m_col6 = st.columns(6)
    
    # Format duration in mm:ss.ms
    mins = int(duration // 60)
    secs = duration % 60
    duration_str = f"{mins:02d}:{secs:05.2f}" if mins > 0 else f"{secs:.2f}s"

    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⏱️ Duration</div>
            <div class="metric-value">{duration_str}</div>
            <div class="metric-sub">{duration:.3f} total sec</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ Sample Rate</div>
            <div class="metric-value">{sr / 1000:.1f} <span style="font-size: 1rem;">kHz</span></div>
            <div class="metric-sub">{sr:,} samples/s</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🎚️ Channels</div>
            <div class="metric-value">{'Stereo (2)' if channels == 2 else 'Mono (1)'}</div>
            <div class="metric-sub">{channels} audio channel{'s' if channels > 1 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📈 Peak Frequency</div>
            <div class="metric-value">{peak_freq:.1f} <span style="font-size: 1rem;">Hz</span></div>
            <div class="metric-sub">Dominant FFT pitch</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">🔊 Peak Amplitude</div>
            <div class="metric-value">{dbfs_val:.1f} <span style="font-size: 1rem;">dBFS</span></div>
            <div class="metric-sub">Max: {peak_val:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">⚡ RMS Energy</div>
            <div class="metric-value">{rms_val:.3f}</div>
            <div class="metric-sub">{20*np.log10(rms_val+1e-12):.1f} dB RMS</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br/>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Visualizations Tabs
    # -------------------------------------------------------------------------
    tab_wave, tab_fft, tab_spec, tab_details = st.tabs([
        "🌊 Waveform (Time Domain)",
        "📊 Frequency Spectrum (FFT)",
        "🌈 Spectrogram (STFT)",
        "📋 Detailed Audio Statistics"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Waveform Plot
    # -------------------------------------------------------------------------
    with tab_wave:
        st.markdown("#### Audio Waveform (Amplitude vs. Time)")
        
        # Downsample for rendering efficiency if total samples > 100,000
        time_axis = np.linspace(0, duration, n_samples)
        
        max_plot_points = 60000
        if n_samples > max_plot_points:
            step = n_samples // max_plot_points
            plot_time = time_axis[::step]
            if channels == 1:
                plot_y = mono_signal[::step]
            else:
                plot_y = y[:, ::step]
        else:
            plot_time = time_axis
            plot_y = y if channels > 1 else mono_signal

        if channels == 2:
            fig_wave, (ax_l, ax_r) = plt.subplots(2, 1, figsize=(12, 5.5), sharex=True, sharey=True)
            
            # Left Channel
            ax_l.plot(plot_time, plot_y[0], color='#00d2ff', linewidth=0.8, alpha=0.9, label='Left Channel')
            ax_l.axhline(0, color='#485460', linestyle='--', linewidth=0.6)
            ax_l.set_ylabel("Amplitude (Left)", fontsize=10)
            ax_l.set_ylim([-1.05, 1.05])
            ax_l.legend(loc='upper right', facecolor='#151A22', edgecolor='#30363D', labelcolor='#C9D1D9')
            
            # Right Channel
            ax_r.plot(plot_time, plot_y[1], color='#38ef7d', linewidth=0.8, alpha=0.9, label='Right Channel')
            ax_r.axhline(0, color='#485460', linestyle='--', linewidth=0.6)
            ax_r.set_ylabel("Amplitude (Right)", fontsize=10)
            ax_r.set_xlabel("Time (seconds)", fontsize=11)
            ax_r.set_ylim([-1.05, 1.05])
            ax_r.legend(loc='upper right', facecolor='#151A22', edgecolor='#30363D', labelcolor='#C9D1D9')
            
            apply_plot_theme(fig_wave, [ax_l, ax_r])
            fig_wave.tight_layout()
            st.pyplot(fig_wave, clear_figure=True)
            plt.close(fig_wave)
        else:
            fig_wave, ax_wave = plt.subplots(figsize=(12, 4.2))
            
            # Mono Channel
            ax_wave.plot(plot_time, plot_y, color='#00d2ff', linewidth=0.8, alpha=0.9, label='Audio Signal')
            
            # Calculate and plot moving RMS envelope
            envelope_window = max(int(sr * 0.05), 1)
            sq_signal = mono_signal ** 2
            rms_envelope = np.sqrt(np.convolve(sq_signal, np.ones(envelope_window)/envelope_window, mode='same'))
            if n_samples > max_plot_points:
                rms_plot = rms_envelope[::step]
            else:
                rms_plot = rms_envelope
                
            ax_wave.plot(plot_time, rms_plot, color='#ff9ff3', linewidth=1.2, linestyle='-', alpha=0.85, label='RMS Envelope')
            ax_wave.plot(plot_time, -rms_plot, color='#ff9ff3', linewidth=1.2, linestyle='-', alpha=0.85)
            
            ax_wave.axhline(0, color='#485460', linestyle='--', linewidth=0.6)
            ax_wave.set_ylabel("Normalized Amplitude", fontsize=11)
            ax_wave.set_xlabel("Time (seconds)", fontsize=11)
            ax_wave.set_ylim([-1.05, 1.05])
            ax_wave.legend(loc='upper right', facecolor='#151A22', edgecolor='#30363D', labelcolor='#C9D1D9')
            
            apply_plot_theme(fig_wave, [ax_wave])
            fig_wave.tight_layout()
            st.pyplot(fig_wave, clear_figure=True)
            plt.close(fig_wave)

    # -------------------------------------------------------------------------
    # TAB 2: Frequency Spectrum (FFT)
    # -------------------------------------------------------------------------
    with tab_fft:
        st.markdown("#### Fast Fourier Transform (FFT) Frequency Spectrum")
        
        # Filter frequency display range
        freq_mask = freqs <= max_fft_freq
        disp_freqs = freqs[freq_mask]
        
        if fft_scale == "Decibels (dBFS Normalized)":
            disp_mag = mag_db[freq_mask]
            y_label = "Magnitude (dB)"
            y_min = max(np.min(disp_mag), -90.0)
            y_max = 5.0
        else:
            disp_mag = mag_linear[freq_mask]
            y_label = "Linear Magnitude"
            y_min = 0.0
            y_max = np.max(disp_mag) * 1.15 if np.max(disp_mag) > 0 else 1.0

        fig_fft, ax_fft = plt.subplots(figsize=(12, 4.6))
        
        # Plot spectrum line and gradient fill
        ax_fft.plot(disp_freqs, disp_mag, color='#38ef7d', linewidth=1.1, label='FFT Spectrum')
        ax_fft.fill_between(disp_freqs, y_min, disp_mag, color='#38ef7d', alpha=0.25)
        
        # Highlight Dominant Peak
        if peak_freq <= max_fft_freq:
            peak_display_val = 0.0 if fft_scale == "Decibels (dBFS Normalized)" else peak_mag
            ax_fft.plot(peak_freq, peak_display_val, 'ro', markersize=7, label=f'Peak: {peak_freq:.1f} Hz')
            ax_fft.annotate(
                f'Peak: {peak_freq:.1f} Hz',
                xy=(peak_freq, peak_display_val),
                xytext=(peak_freq + (max_fft_freq * 0.03), peak_display_val - (10 if "Decibels" in fft_scale else (y_max * 0.1))),
                arrowprops=dict(facecolor='#ff6b6b', shrink=0.08, width=1.2, headwidth=6),
                color='#ff6b6b',
                fontweight='bold',
                fontsize=9
            )

        ax_fft.set_xlim([0, max_fft_freq])
        ax_fft.set_ylim([y_min, y_max])
        ax_fft.set_xlabel("Frequency (Hz)", fontsize=11)
        ax_fft.set_ylabel(y_label, fontsize=11)
        ax_fft.legend(loc='upper right', facecolor='#151A22', edgecolor='#30363D', labelcolor='#C9D1D9')
        
        apply_plot_theme(fig_fft, [ax_fft])
        fig_fft.tight_layout()
        st.pyplot(fig_fft, clear_figure=True)
        plt.close(fig_fft)

        # Acoustic Frequency Bands Breakdown
        st.markdown("##### 🎶 Frequency Band Energy Distribution")
        band_energies = calculate_frequency_bands(freqs, mag_linear)
        
        b_cols = st.columns(len(band_energies))
        for idx, (b_name, b_pct) in enumerate(band_energies.items()):
            with b_cols[idx]:
                st.metric(label=b_name.split(" ")[0], value=f"{b_pct:.1f}%")
                st.progress(min(max(b_pct / 100.0, 0.0), 1.0))

    # -------------------------------------------------------------------------
    # TAB 3: Spectrogram (STFT)
    # -------------------------------------------------------------------------
    with tab_spec:
        st.markdown("#### Short-Time Fourier Transform (STFT) Spectrogram")
        
        with st.spinner("Computing spectrogram..."):
            # Compute STFT
            n_fft = 2048
            hop_length = 512
            D = librosa.stft(mono_signal, n_fft=n_fft, hop_length=hop_length)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            fig_spec, ax_spec = plt.subplots(figsize=(12, 4.6))
            
            # Use librosa.display.specshow
            img = librosa.display.specshow(
                S_db,
                sr=sr,
                hop_length=hop_length,
                x_axis='time',
                y_axis='hz',
                cmap=spectrogram_cmap,
                ax=ax_spec
            )
            
            ax_spec.set_ylim([0, max_fft_freq])
            ax_spec.set_xlabel("Time (seconds)", fontsize=11)
            ax_spec.set_ylabel("Frequency (Hz)", fontsize=11)
            
            cbar = fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
            cbar.ax.yaxis.set_tick_params(color='#8B949E')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#8B949E')
            cbar.set_label('Magnitude (dB)', color='#C9D1D9', fontsize=10)
            
            apply_plot_theme(fig_spec, [ax_spec])
            fig_spec.tight_layout()
            st.pyplot(fig_spec, clear_figure=True)
            plt.close(fig_spec)

    # -------------------------------------------------------------------------
    # TAB 4: Detailed Audio Statistics
    # -------------------------------------------------------------------------
    with tab_details:
        st.markdown("#### 📋 Detailed Technical Audio Attributes")
        
        # Additional librosa feature metrics
        zero_crossings = np.sum(librosa.zero_crossings(mono_signal, pad=False))
        zcr_rate = zero_crossings / len(mono_signal)
        
        # Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=mono_signal, sr=sr)
        mean_centroid = np.mean(cent)
        
        # Spectral Rolloff (85%)
        rolloff = librosa.feature.spectral_rolloff(y=mono_signal, sr=sr, roll_percent=0.85)
        mean_rolloff = np.mean(rolloff)
        
        # Dynamic Range
        dynamic_range_db = 20 * np.log10((peak_val / (np.min(np.abs(mono_signal[mono_signal != 0])) + 1e-12)))

        col_d1, col_d2 = st.columns(2)
        
        with col_d1:
            st.markdown(f"""
            - **Total Samples**: `{n_samples:,}`
            - **Duration**: `{duration:.4f}` seconds ({duration_str})
            - **Sample Rate**: `{sr:,}` Hz ({sr/1000:.1f} kHz)
            - **Audio Channels**: `{channels}` ({'Stereo' if channels == 2 else 'Mono'})
            - **Dominant Frequency**: `{peak_freq:.2f}` Hz
            - **Max Peak Amplitude**: `{peak_val:.5f}` ({dbfs_val:.2f} dBFS)
            """)

        with col_d2:
            st.markdown(f"""
            - **RMS Amplitude**: `{rms_val:.5f}` ({20*np.log10(rms_val+1e-12):.2f} dB)
            - **Zero Crossing Count**: `{zero_crossings:,}` (Rate: `{zcr_rate:.4f}`)
            - **Spectral Centroid (Mean)**: `{mean_centroid:.2f}` Hz
            - **Spectral Rolloff (85%)**: `{mean_rolloff:.2f}` Hz
            - **Estimated Dynamic Range**: `{dynamic_range_db:.2f}` dB
            - **Nyquist Frequency Limit**: `{sr / 2.0:,}` Hz
            """)


if __name__ == "__main__":
    main()
