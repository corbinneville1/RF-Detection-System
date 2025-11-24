# RF-Detection-System

Headless Pi 4 + B200 RF early-warning rig with multi-band scan and spoken alerts.

`rf_watcher` is a headless RF “spotter” for a Raspberry Pi 4B and Ettus B200. It continuously scans multiple bands (ATC VHF, 2m amateur, VHF high, FRS/GMRS, and a 1090 MHz ADS-B sector) and runs FFT-based energy detection to find signals that rise above the local noise floor. When activity is detected, the Pi uses offline text-to-speech to speak short alerts directly into a wired headset, giving hands-free RF situational awareness from a belt or sling-bag rig.

The system is designed to run completely offline on battery power (PiSugar S-Plus 5000 mAh), automatically starting on boot via systemd. It supports per-band thresholds, frequency bucketing to prevent multiple alerts from one carrier, per-frequency cooldowns, and notches for known constant tones (e.g., 146.000 MHz). In its current configuration it achieves roughly 2–2.5 hours of continuous scan time per charge while sweeping all configured bands several times per second.

## System dependencies

This project uses the Ettus UHD drivers and Python bindings. Install:

```bash
sudo apt update
sudo apt install python3-uhd uhd-host
'''

## Python dependencies

Install the Python packages with:

```bash
pip install -r requirements.txt
'''
