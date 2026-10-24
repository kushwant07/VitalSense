import numpy as np
from scipy import signal
from scipy.signal import find_peaks, welch, butter, filtfilt, savgol_filter

_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))

def bandpass_filter(signal_data, fs, low=0.7, high=3.0, order=3):
    nyq = 0.5 * fs 
    if high >= nyq: high = nyq - 0.1
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, signal_data)

def clean_signal(wave, fs):
    wave = signal.detrend(wave)
    # Use wider band for general cleaning (up to 210 BPM)
    wave = bandpass_filter(wave, fs, low=0.7, high=3.5, order=4)
    
    # Amplitude normalization (Sliding RMS)
    win = int(5.0 * fs)
    if win < 1: win = 1
    pad = np.pad(wave, (win//2, win//2), mode='edge')
    rms = np.array([np.sqrt(np.mean(pad[i:i+win]**2)) for i in range(len(wave))])
    wave = wave / (rms + 1e-6)
    
    # Z-score normalization
    return (wave - np.mean(wave)) / (np.std(wave) + 1e-6)

def refine_peaks_interp(sig, peaks):
    refined_peaks = []
    for p in peaks:
        if p < 1 or p >= len(sig) - 1:
            refined_peaks.append(float(p))
            continue
            
        alpha = sig[p-1]
        beta = sig[p]
        gamma = sig[p+1]
        
        denom = (alpha - 2*beta + gamma)
        if denom == 0: 
            refined_peaks.append(float(p))
            continue
            
        delta = 0.5 * (alpha - gamma) / denom
        refined_peaks.append(p + delta)
        
    return np.array(refined_peaks)

def calculate_baevsky_stress(rr_ms):
    if len(rr_ms) < 10: return 0.0
    
    bin_size = 50 
    bins = np.arange(min(rr_ms), max(rr_ms) + bin_size, bin_size)
    hist, edges = np.histogram(rr_ms, bins=bins)
    
    if len(hist) == 0: return 0.0
    
    max_bin_idx = np.argmax(hist)
    Mo = (edges[max_bin_idx] + edges[max_bin_idx+1]) / 2 / 1000.0 
    
    AMo = (hist[max_bin_idx] / len(rr_ms)) * 100.0
    MxDM = (np.max(rr_ms) - np.min(rr_ms)) / 1000.0 
    
    if MxDM == 0 or Mo == 0: return 0.0
    
    SI = AMo / (2 * Mo * MxDM)
    return SI

def compute_hrv_features(ppg, fs):
    peaks, _ = find_peaks(ppg, distance=fs*0.5)
    
    if len(peaks) < 2: 
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    true_peaks = refine_peaks_interp(ppg, peaks)
    rr = np.diff(true_peaks) / fs * 1000 
    
    sdnn = np.std(rr)
    rmssd = np.sqrt(np.mean(np.diff(rr)**2))
    pnn50 = np.sum(np.abs(np.diff(rr)) > 50) / len(rr) * 100
    
    n_seg = min(len(rr), 256)
    if n_seg < 2: return sdnn, rmssd, pnn50, 0.0, 0.0
    
    f, p = welch(rr/1000, fs=4.0, nperseg=n_seg)
    lf = _trapz(p[(f>=0.04)&(f<0.15)], f[(f>=0.04)&(f<0.15)])
    hf = _trapz(p[(f>=0.15)&(f<0.4)], f[(f>=0.15)&(f<0.4)])
    ratio = lf/(hf+1e-6)
    
    si = calculate_baevsky_stress(rr)
    
    return sdnn, rmssd, pnn50, ratio, si

def run_pos_algorithm(r, g, b, fs):
    win = int(1.6 * fs)
    l = len(r)
    h = np.zeros(l)
    for i in range(0, l - win, 1):
        seg_r = r[i:i+win]; seg_g = g[i:i+win]; seg_b = b[i:i+win]
        cn_r = seg_r / (np.mean(seg_r) + 1e-6)
        cn_g = seg_g / (np.mean(seg_g) + 1e-6)
        cn_b = seg_b / (np.mean(seg_b) + 1e-6)
        s1 = cn_g - cn_b
        s2 = cn_g + cn_b - 2 * cn_r
        alpha = np.std(s1) / (np.std(s2) + 1e-6)
        h_seg = s1 + alpha * s2
        h[i + win//2] = h_seg[win//2]
        
    # Fill boundary edges with reflection to avoid zero-step ringing in bandpass
    half = win // 2
    if half > 0 and l > half * 2:
        h[:half] = h[half:half*2][::-1]
        h[-half:] = h[-half*2:-half][::-1]
        
    return clean_signal(h, fs)

def get_dominant_freq(sig, fs):
    if len(sig) < 16 or fs <= 0:
        return 0.0

    # 1. Bandpass filter to physiological HR range [48, 190] BPM (0.8 - 3.2 Hz)
    try:
        filt = bandpass_filter(sig, fs, low=0.75, high=3.2, order=3)
        filt = (filt - np.mean(filt)) / (np.std(filt) + 1e-6)
    except Exception:
        filt = sig - np.mean(sig)

    N = len(filt)

    # Normalized Autocorrelation helper (unbiased)
    def _norm_autocorr(x, lag):
        if lag <= 0 or lag >= N:
            return 0.0
        x1 = x[:N - lag]
        x2 = x[lag:]
        denom = np.sqrt(np.sum(x1 ** 2) * np.sum(x2 ** 2)) + 1e-8
        return float(np.sum(x1 * x2) / denom)

    # 2. Frequency-domain Welch PSD
    n_seg = min(len(filt), 256)
    if n_seg < 32:
        n_seg = len(filt)
    f, p = welch(filt, fs=fs, nperseg=n_seg, nfft=2048)
    mask = (f >= 0.75) & (f <= 3.1)
    if not np.any(mask):
        return 75.0

    f_sub = f[mask]
    p_sub = p[mask]

    freq_step = f[1] - f[0]
    min_dist = max(1, int(0.20 / freq_step))  # ~12 BPM separation
    peaks, _ = find_peaks(p_sub, height=np.max(p_sub) * 0.18, distance=min_dist)

    if len(peaks) == 0:
        return float(f_sub[np.argmax(p_sub)] * 60.0)

    top_idx = peaks[np.argmax(p_sub[peaks])]
    top_bpm = f_sub[top_idx] * 60.0
    p_max = p_sub[top_idx]

    # Rule 1: Elevated 2nd harmonic suppression (>= 130 BPM).
    # If dominant peak is >= 130 BPM and a subharmonic in resting range [55, 95] exists with >= 20% power:
    if top_bpm >= 130.0:
        half_bpm = top_bpm / 2.0
        for cp in peaks:
            bpm = f_sub[cp] * 60.0
            if 55.0 <= bpm <= 95.0 and abs(bpm - half_bpm) / half_bpm <= 0.12:
                if p_sub[cp] >= 0.20 * p_max:
                    return float(bpm)

    # Rule 2: Scoring among prominent peaks (>= 60% of max power).
    # Prevents weak noise sidebands (<60% power) from ever beating the dominant cardiac pulse.
    candidates = []
    for cp in peaks:
        bpm = f_sub[cp] * 60.0
        rel_p = p_sub[cp] / p_max
        if rel_p >= 0.60:
            lag = int(round(fs / f_sub[cp]))
            ac = max(0.0, _norm_autocorr(filt, lag))
            score = rel_p * (1.0 + 0.40 * ac)
            candidates.append((bpm, score))

    if candidates:
        best_bpm = max(candidates, key=lambda x: x[1])[0]
        return float(best_bpm)

    return float(top_bpm)

# --- NEW BREATHING LOGIC (AI Output Processing) ---
def extract_breathing_rate_from_envelope(envelope, fs):
    """
    Computes BR from the respiration envelope (AI Output) using Welch PSD.
    Band: 0.12 Hz (7.2 BPM) - 0.45 Hz (27 BPM)
    """
    if len(envelope) < fs * 2: return 15.0 # Not enough data
    
    # 1. Smooth the envelope
    try:
        win_len = 31
        if len(envelope) < win_len: win_len = len(envelope) // 2 * 2 + 1
        if win_len < 3: win_len = 3
        smooth_env = savgol_filter(envelope, window_length=win_len, polyorder=3)
    except:
        smooth_env = envelope

    # 2. Welch PSD
    f, pxx = welch(smooth_env, fs=fs, nperseg=min(len(smooth_env), 512), nfft=4096)
    
    # 3. Mask for valid physiological breathing range
    RESP_LOW = 0.13   # 7.8 BrPM
    RESP_HIGH = 0.45  # 27.0 BrPM
    mask = (f >= RESP_LOW) & (f <= RESP_HIGH)
    
    if not np.any(mask) or np.sum(pxx[mask]) == 0:
        return 15.0
        
    # 4. Peak Finding
    best_freq = f[mask][np.argmax(pxx[mask])]
    br_bpm = best_freq * 60.0
    
    return float(round(br_bpm, 1))

# --- FALLBACK LOGIC ---
def predict_breathing_heuristic(signal_data, fs):
    """
    Extracts breathing rate directly from respiration bandpass.
    Excludes low-frequency 1/f noise below 0.13 Hz (7.8 BrPM).
    """
    if len(signal_data) < fs * 5: return 15.0 
    
    # 1. Filter to Respiratory Band (0.13 - 0.45 Hz = 7.8 - 27 BrPM)
    try:
        nyq = 0.5 * fs
        low_cut = 0.13 / nyq
        high_cut = min(0.45 / nyq, 0.95)
        b, a = butter(2, [low_cut, high_cut], btype='band')
        resp_sig = filtfilt(b, a, signal_data)
    except Exception:
        return 15.0

    # 2. Welch PSD
    n_seg = min(len(resp_sig), 512)
    f, p = welch(resp_sig, fs=fs, nperseg=n_seg, nfft=4096)
    
    # 3. Peak Finding
    mask = (f >= 0.13) & (f <= 0.45)
    
    if not np.any(mask) or np.sum(p[mask]) == 0:
        return 15.0 
    
    best_freq = f[mask][np.argmax(p[mask])]
    return float(round(best_freq * 60.0, 1))