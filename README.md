# Type III Solar Radio Burst Analysis Toolkit

**SOLER WP2 – Radio Data Analysis Methods**

A Python toolkit and Jupyter‐notebook tutorial for loading, visualizing, and analysing Type III solar radio bursts recorded by the NenuFAR low-frequency array (10–85 MHz).  
---

**Developed by:** Jinge ZHANG, Paris Observatory (Meudon Site)

**My e-mail::** Jinge.Zhang@obspm.fr

---

## 📖 Overview

This repository provides:

1. **Data‐loading routines** (`combine_spectra.py`)  
   • Fetch and combine low- and high-band NenuFAR `.spectra` files  
   • Rebin in time/frequency; apply sliding‐window normalization  
   • Write out wide-band HDF5 dynamic‐spectrum files per Stokes parameter  
   • “Quick-look” plotting of the resulting dynamic spectra  

2. **Region‐of‐Interest detection** (`window_detection.py`)  
   • Slice out a user-defined time/frequency window from your dynamic spectrum  
   • Compute per-channel peak times and frequencies    

3. **Time-Profile Fitting** (`T3_time_profile.py`)  
   • Fit each ROI light-curve (per-channel or band-averaged) with multiple models:  
     – Exponential decay
     – Symmetric Gaussian 
     – Bi-Gaussian (different rise/decay widths)  
     – Chrysaphi (asymmetric “alpha-function” profile)  
   • Automatically select the best-fit model by χ² test
   • Compute start, peak, and end times from FWHM or 1/e width criteria  
   • Report both rise and decay durations (time from start→peak and peak→end)  

4. **Burst Velocity Analysis** (`analysis_velocity.py`)  
   • Extract the time-frequency trajectory of burst's flux
   • Compute average linear drift (df/dt) and instantaneous drift 
   • Convert drift rates into electron-beam velocities using user-selectable coronal density models and emission mechanisms, estimating speeds at the burst’s start, peak, and end phases.

5. **Jupyter Notebook Tutorial**  (` T3RadioBurst_Tool_HandsOn.ipynb`) 
   • Step-by-step notebook that walks through Parts 1–3:  
     - Part 1: Load & visualize NenuFAR dynamic spectra  
     - Part 2: Define burst Regions of Interest (ROIs)  
     - Part 3: Time-profile fitting & exciter velocity determination  

---

## ⚙️ Installation

1. **Clone this repo**  
   ```bash
   git clone https://github.com/your-username/T3BurstTool.git
   cd T3BurstTool
