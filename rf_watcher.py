#!/usr/bin/env python3
"""
rf_watcher.py

Headless RF watcher for Raspberry Pi 4 + Ettus B200.

- Scans configured wide bands using FFT + noise-floor threshold.
- Fast-hops FRS/GMRS channels as a separate scan mode.
- Speaks alerts via espeak-ng when strong signals are detected.

Run:  python3 rf_watcher.py
"""

import time
import subprocess
from typing import List, Dict, Tuple, Optional

import numpy as np
import uhd  # from python3-uhd


# -----------------------------
# GLOBAL CONFIG
# -----------------------------

# Threshold above noise floor (in dB) to declare "activity"
THRESHOLD_WIDE_DB = 26.0 # for WATCH_BANDS was 26.0
THRESHOLD_FRS_DB = 28.0 #for FRS/GMRS channel scan was 24.0

# Seconds before repeating an alert for a given freq/channel
ALERT_HOLDOFF_SEC = 60.0 # was 15.0

# FFT size (power-of-two; 1024/2048 are OK for Pi)
FFT_SIZE = 2048

# USRP RX gain
DEFAULT_GAIN = 30.0 # was 40.0

# Run both modes (your “end state”)
ENABLE_WIDE_BANDS = True
ENABLE_FRS_GMRS_SCAN = True

# FRS/GMRS scan parameters
FRS_GMRS_SCAN_SAMPLERATE = 250e3     # 250 kS/s
FRS_GMRS_CHANNEL_BW_HZ = 25e3       # +/- 12.5 kHz
FRS_GMRS_DWELL_SEC = 0.03           # ~30 ms per channel


# -----------------------------
# CONFIG: WIDE BANDS
# -----------------------------

WATCH_BANDS: List[Dict] = [
    # A couple of example ATC slices (you can add more later)
    {
        "name": "ATC_118",
        "center_hz": 119e6,
        "sample_rate_hz": 2e6,
        "watch_range_hz": (118e6, 120e6),
        "mode": "AM",
    },
    {
        "name": "ATC_120",
        "center_hz": 121e6,
        "sample_rate_hz": 2e6,
        "watch_range_hz": (120e6, 122e6),
        "mode": "AM",
    },

    # 2 Meter HAM Band
    {
        "name": "2M",
        "center_hz": 146e6,              # center of 144–148 MHz
        "sample_rate_hz": 4e6,           # covers 144–148 MHz
        "watch_range_hz": (144e6, 148e6),
        "mode": "FM",
    },

    # VHF high 162–166 MHz
    {
        "name": "VHF_HIGH",
        "center_hz": 164e6,
        "sample_rate_hz": 4e6,  # drop to 2e6 if Pi struggles
        "watch_range_hz": (162e6, 166e6),
        "mode": "mixed",
    },

    # ADS-B 1090 MHz
    {
        "name": "ADS_B",
        "center_hz": 1090e6,
        "sample_rate_hz": 2e6,
        "watch_range_hz": (1089e6, 1091e6),
        "mode": "pulse",
    },

    # Wide chunk around FRS/GMRS (in addition to channel scan)
    {
        "name": "FRS_GMRS",
        "center_hz": 462.65e6,
        "sample_rate_hz": 3.2e6,   # covers ~461.15–464.15 MHz it was 3e6
        "watch_range_hz": (462.4e6, 462.8e6),
        "mode": "FM",
    },
]


# -----------------------------
# CONFIG: CHANNEL LISTS
# -----------------------------

FRS_GMRS_CHANNELS = [
    {"name": "FRS GMRS 1", "freq_hz": 462.5625e6},
    {"name": "FRS GMRS 2", "freq_hz": 462.5875e6},
    {"name": "FRS GMRS 3", "freq_hz": 462.6125e6},
    {"name": "FRS GMRS 4", "freq_hz": 462.6375e6},
    {"name": "FRS GMRS 5", "freq_hz": 462.6625e6},
    {"name": "FRS GMRS 6", "freq_hz": 462.6875e6},
    {"name": "FRS GMRS 7", "freq_hz": 462.7125e6},
    {"name": "GMRS 15",    "freq_hz": 462.5500e6},
    {"name": "GMRS 16",    "freq_hz": 462.5750e6},
    {"name": "GMRS 17",    "freq_hz": 462.6000e6},
    {"name": "GMRS 18",    "freq_hz": 462.6250e6},
    {"name": "GMRS 19",    "freq_hz": 462.6500e6},
    {"name": "GMRS 20",    "freq_hz": 462.6750e6},
    {"name": "GMRS 21",    "freq_hz": 462.7000e6},
    {"name": "GMRS 22",    "freq_hz": 462.7250e6},
]

WX_CHANNELS = [
    {"name": "WX1", "freq_hz": 162.400e6},
    {"name": "WX2", "freq_hz": 162.425e6},
    {"name": "WX3", "freq_hz": 162.450e6},
    {"name": "WX4", "freq_hz": 162.475e6},
    {"name": "WX5", "freq_hz": 162.500e6},
    {"name": "WX6", "freq_hz": 162.525e6},
    {"name": "WX7", "freq_hz": 162.550e6},
]

SPECIAL_CHANNEL_GROUPS = {
    "FRS_GMRS": FRS_GMRS_CHANNELS,
    "WX": WX_CHANNELS,
}


# -----------------------------
# TEXT-TO-SPEECH
# -----------------------------

def speak(text: str) -> None:
    print(f"[TTS] {text}")
    try:
        wav_path = "/tmp/rf_watcher_tts.wav"

        # 1) Synthesize text to a WAV file
        subprocess.run(
            ["/usr/bin/espeak-ng", "-a", "200", "-w", wav_path, text],
            check=False
        )

        # 2) Play the WAV via ALSA (same path that worked in your test)
        subprocess.run(
            ["/usr/bin/aplay", wav_path],
            check=False
        )
    except Exception as e:
        print(f"[TTS ERROR] {e}")


# -----------------------------
# HELPERS
# -----------------------------

def format_freq_for_speech(freq_hz: float, decimals: int = 4) -> str:
    """
    462.6125 MHz -> "four six two point six one two five megahertz"
    """
    mhz = freq_hz / 1e6
    fmt = f"{{:.{decimals}f}} megahertz"
    return fmt.format(mhz).replace(".", " point ")


def nearest_special_channel(freq_hz: float, max_delta_hz: float = 5e3) -> Tuple[Optional[str], Optional[float]]:
    closest_name = None
    closest_freq = None
    closest_delta = float("inf")

    for chans in SPECIAL_CHANNEL_GROUPS.values():
        for ch in chans:
            delta = abs(ch["freq_hz"] - freq_hz)
            if delta < closest_delta and delta <= max_delta_hz:
                closest_delta = delta
                closest_name = ch["name"]
                closest_freq = ch["freq_hz"]

    return closest_name, closest_freq


def estimate_noise_floor_db(power_db: np.ndarray) -> float:
    """20th percentile as noise floor."""
    return np.percentile(power_db, 20)


# -----------------------------
# USRP / B200 SETUP
# -----------------------------

def make_usrp() -> Optional[uhd.usrp.MultiUSRP]:
    """
    Create and basic-configure the USRP (Ettus B200).
    Returns None if no device is found.
    """
    try:
        usrp = uhd.usrp.MultiUSRP()  # default device addr
    except Exception as e:
        print(f"[ERROR] Unable to create USRP device: {e}")
        return None

    # Try to use TX/RX antenna; ignore if not present
    try:
        usrp.set_rx_antenna("TX/RX")
    except RuntimeError:
        pass

    return usrp


# -----------------------------
# WIDE-BAND SCAN
# -----------------------------

def scan_band(
    usrp: uhd.usrp.MultiUSRP,
    band_cfg: Dict,
    last_alerts: Dict[str, float],
) -> None:
    name = band_cfg["name"]
    center = float(band_cfg["center_hz"])
    samp_rate = float(band_cfg["sample_rate_hz"])
    low, high = band_cfg["watch_range_hz"]

    usrp.set_rx_rate(samp_rate)
    usrp.set_rx_freq(center)
    usrp.set_rx_gain(DEFAULT_GAIN)

    # Give the tuner a moment to settle
    time.sleep(0.05)

    # Create RX streamer
    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_stream = usrp.get_rx_stream(st_args)

    # Tell UHD we want exactly FFT_SIZE samples now
    stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
    stream_cmd.num_samps = FFT_SIZE
    stream_cmd.stream_now = True
    stream_cmd.time_spec = uhd.types.TimeSpec(0.0)
    rx_stream.issue_stream_cmd(stream_cmd)

    # Allocate buffer and receive
    buff = np.zeros(FFT_SIZE, dtype=np.complex64)
    md = uhd.types.RXMetadata()
    num_rx = rx_stream.recv(buff, md, timeout=0.2)
    if num_rx <= 0 or md.error_code != uhd.types.RXMetadataErrorCode.none:
        print(f"[WARN] RX issue on band {name}: {md.error_code}")
        return


    spectrum = np.fft.fftshift(np.fft.fft(buff))
    power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)

    noise_floor_db = estimate_noise_floor_db(power_db)

    freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1.0 / samp_rate))
    freqs_hz = center + freqs

    now = time.time()

    for f_hz, p_db in zip(freqs_hz, power_db):
        if f_hz < low or f_hz > high:
            continue

        # Ignore a constant carrier near 146.000 MHz in the 2m band
        if name == "2M" and abs(f_hz - 146.000e6) < 10e3:  # +/- 5 kHz window
            continue

        if p_db - noise_floor_db < THRESHOLD_WIDE_DB:
            continue

        alert_freq_key = f"{name}:{int(round(f_hz / 100.0)) * 100}"
        last_time = last_alerts.get(alert_freq_key, 0.0)
        if (now - last_time) < ALERT_HOLDOFF_SEC:
            continue

        chan_name, chan_freq = nearest_special_channel(f_hz, max_delta_hz=7.5e3)
        if chan_name and chan_freq:
            speech_freq = format_freq_for_speech(chan_freq)
            msg = f"{chan_name} traffic at {speech_freq}" #"{chan_name} active on {speech_freq}"
        else:
            speech_freq = format_freq_for_speech(f_hz, decimals=3)
            msg = f"{name} activity {speech_freq}" #"Strong signal in band {name} at {speech_freq}"

        speak(msg)
        last_alerts[alert_freq_key] = now


# -----------------------------
# FRS/GMRS CHANNEL SCAN
# -----------------------------

def scan_frs_gmrs_channels(
    usrp: uhd.usrp.MultiUSRP,
    last_alerts: Dict[str, float],
) -> None:
    samp_rate = float(FRS_GMRS_SCAN_SAMPLERATE)

    usrp.set_rx_rate(samp_rate)
    usrp.set_rx_gain(DEFAULT_GAIN)

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    rx_stream = usrp.get_rx_stream(st_args)
    buff = np.zeros(FFT_SIZE, dtype=np.complex64)

    for ch in FRS_GMRS_CHANNELS:
        ch_name = ch["name"]
        ch_freq = float(ch["freq_hz"])

        usrp.set_rx_freq(ch_freq)
        time.sleep(FRS_GMRS_DWELL_SEC)

        # Request FFT_SIZE samples at this channel
        stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
        stream_cmd.num_samps = FFT_SIZE
        stream_cmd.stream_now = True
        stream_cmd.time_spec = uhd.types.TimeSpec(0.0)
        rx_stream.issue_stream_cmd(stream_cmd)

        md = uhd.types.RXMetadata()
        num_rx = rx_stream.recv(buff, md, timeout=0.2)
        if num_rx <= 0 or md.error_code != uhd.types.RXMetadataErrorCode.none:
            # Just skip this channel on timeout/error
            # print(f"[WARN] RX issue on FRS channel {ch_name}: {md.error_code}")
            continue


        spectrum = np.fft.fftshift(np.fft.fft(buff))
        power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-12)

        noise_floor_db = estimate_noise_floor_db(power_db)

        freqs = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1.0 / samp_rate))
        freqs_hz = ch_freq + freqs

        half_bw = FRS_GMRS_CHANNEL_BW_HZ / 2.0
        mask = np.abs(freqs_hz - ch_freq) <= half_bw
        if not np.any(mask):
            continue

        channel_power = power_db[mask]
        peak_db = float(np.max(channel_power))
        delta_db = peak_db - noise_floor_db

        if delta_db < THRESHOLD_FRS_DB:
            continue

        now = time.time()
        alert_key = f"FRS:{ch_name}"
        last_time = last_alerts.get(alert_key, 0.0)
        if (now - last_time) < ALERT_HOLDOFF_SEC:
            continue

        speech_freq = format_freq_for_speech(ch_freq)
        msg = f"{ch_name} active on {speech_freq}"
        speak(msg)
        last_alerts[alert_key] = now


# -----------------------------
# MAIN
# -----------------------------

def main():
    print("[INFO] Starting RF watcher...")
    usrp = make_usrp()
    if usrp is None:
        print("[FATAL] No USRP/B200 detected. Plug it in and try again.")
        return

    last_alerts: Dict[str, float] = {}
    time.sleep(1.0)

    while True:
        if ENABLE_WIDE_BANDS:
            for band in WATCH_BANDS:
                try:
                    scan_band(usrp, band, last_alerts)
                except KeyboardInterrupt:
                    print("\n[INFO] Stopping watcher (Ctrl+C).")
                    return
                except Exception as e:
                    print(f"[ERROR] While scanning {band['name']}: {e}")

        if ENABLE_FRS_GMRS_SCAN:
            try:
                scan_frs_gmrs_channels(usrp, last_alerts)
            except KeyboardInterrupt:
                print("\n[INFO] Stopping watcher (Ctrl+C).")
                return
            except Exception as e:
                print(f"[ERROR] While FRS/GMRS scan: {e}")

        time.sleep(0.1)


if __name__ == "__main__":
    main()

