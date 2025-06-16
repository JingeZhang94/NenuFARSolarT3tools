import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from scipy import stats, optimize
from scipy.interpolate import splev, splrep
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter, NullFormatter, LogLocator
from scipy.optimize import curve_fit


# —————————————————————————————————————————————————
# 1) Model definitions (copy‐pasted from T3_time_profile_FIT2)
# —————————————————————————————————————————————————

def exponential(t, A, tau, C):
    return A * np.exp(-t / tau) + C

def gaussian(t, A, t0, tau, C):
    return A * np.exp(-0.5 * ((t - t0) / tau)**2) + C

def biGaussian(t, A, t0, tau1, tau2, C):
    dt = t - t0
    sigma = np.where(dt < 0, tau1, tau2)
    return A * np.exp(-0.5 * (dt / sigma)**2) + C

def Chrysaphi(t, A, tau1, tau2, toff, C):
    dt = t - toff
    S = np.full_like(dt, C)
    mask = dt > 0
    S[mask] += A * np.exp(-tau1 / dt[mask] - dt[mask] / tau2)
    return S

# map mode names to functions + parameter lists
MODELS = {
    'exponential': (exponential, ['A','tau','C']),
    'gaussian':    (gaussian,    ['A','t0','tau','C']),
    'biGaussian':  (biGaussian,  ['A','t0','tau1','tau2','C']),
    'Chrysaphi':(Chrysaphi,['A','tau1','tau2','toff','C'])
}



# —————————————————————————————————————————————————
# 2) χ² calculator (unchanged)
# —————————————————————————————————————————————————

def compute_chi2(y, ymod, sigma=None, chi_def='standard', dof=None):
    res = y - ymod
    if sigma is not None:
        res = res / sigma
    chi2 = float(np.nansum(res**2))
    if chi_def == 'reduced':
        if dof is None or dof <= 0:
            raise ValueError("dof>0 required for reduced χ²")
        return chi2 / dof
    return chi2


# —————————————————————————————————————————————————
# 3) Main analyzer
# —————————————————————————————————————————————————

def analyze_burst_velocity(
    burst_data,
    density_model="saito",
    emission_mechanism="F",# "F"=fundamental, "H"=second harmonic
    freq_range=None,
    fit_method="FWHM", #"FWHM" or "1/e"
    y_scale="none",
    fit_mode="none",  # ✅ "none", "single", "split"
    debug=False,        # ✅ Enable/Disable debug plots
    debug_freq_ranges=None,  #None for every channel, and can be multiple ranges, such as [(42, 42.5), (45,46)]
    show_drift_panel=True,
    show_beam_width=False,      
    show_density_models=False
):
    """
    Analyze and visualize the electron beam velocity from a given burst's dynamic spectrum.

    Parameters:
        burst_data (dict): Contains extracted dynamic spectrum, time, frequency, and burst metadata.
        density_model (str): The coronal density model used for distance calculation.
        freq_range (tuple, optional): Custom frequency range (min_freq, max_freq).
        fit_method (str): "FWHM" or "1/e"
        y_scale (str): "log" for logscale or "inverse" for inverse frequency (1/f scale)
        fit_mode (str): "none", "single", "split"
        debug (bool): If True, plot the flux curve and fitted Gaussians.

    Returns:
        None
    """

    burst = burst_data["burst"]
    full_time_mpl = burst_data["full_time_mpl"]
    full_data = burst_data["full_data"]
    roi_data = burst_data["roi_data"]
    roi_time_mpl = burst_data["roi_time_mpl"]
    roi_freq_values = burst_data["roi_freq_values"]

    # --- NEW: if there's a singleton pol‐axis, drop it so roi_data is (ntime, nfreq) ---
    if roi_data.ndim == 3 and roi_data.shape[2] == 1:
        roi_data = roi_data[:, :, 0]

    print(f"\n🔹 Processing Burst {burst['number']}...")
    print(f"   Time Range: {burst['start_time']} to {burst['end_time']}")
    print(f"   Frequency Range: {burst['start_freq']:.2f} MHz to {burst['end_freq']:.2f} MHz")
    print(f"   Extracted Data Shape: {roi_data.shape}")

    # ✅ Apply custom frequency range if provided
    if freq_range is not None:
        min_freq, max_freq = freq_range
        valid_indices = np.where((roi_freq_values >= min_freq) & (roi_freq_values <= max_freq))[0]

        if len(valid_indices) == 0:
            print(f"⚠️ Warning: No valid frequency data in the range {min_freq} - {max_freq} MHz")
            return

        roi_data = roi_data[:, valid_indices]
        roi_freq_values = roi_freq_values[valid_indices]

    # ✅ Convert time to seconds relative to burst start time
    burst_start_time = roi_time_mpl[0]
    burst_end_time = roi_time_mpl[-1]
    peak_times = []
    peak_freqs = []
    start_times = []
    end_times = []
    start_freqs = []
    end_freqs   = []

    for i, freq in enumerate(roi_freq_values):
        flux_values = roi_data[:, i]
        peak_idx = np.argmax(flux_values)
        peak_flux = flux_values[peak_idx]

        # ✅ Record peak time and frequency
        peak_times.append(roi_time_mpl[peak_idx])
        peak_freqs.append(freq)

        # ✅ Set threshold for FWHM or 1/e
        if fit_method == "FWHM":
            threshold_flux = peak_flux / 2
        elif fit_method == "1/e":
            threshold_flux = peak_flux / np.e
        else:
            raise ValueError("Invalid fit_method. Use 'FWHM' or '1/e'.")

        if fit_mode == "none":
            try:
                # --- Baseline subtraction and normalization ---
                baseline = np.min(flux_values)
                flux_corrected = flux_values - baseline
                peak_flux_corrected = flux_values[peak_idx] - baseline
                if peak_flux_corrected <= 0:
                    raise ValueError("Peak flux after baseline subtraction is non-positive.")
                flux_norm = flux_corrected / peak_flux_corrected

                # --- Define the normalized threshold ---
                if fit_method == "FWHM":
                    threshold_norm = 0.5
                elif fit_method == "1/e":
                    threshold_norm = 1/np.e
                else:
                    raise ValueError("Invalid fit_method. Use 'FWHM' or '1/e'.")

                # --- Determine start index: last index before peak where normalized flux falls below threshold ---
                start_idx = peak_idx
                while start_idx > 0:
                    if flux_norm[start_idx] < threshold_norm:
                        break
                    start_idx -= 1

                # --- Determine end index: first index after peak where normalized flux falls below threshold ---
                end_idx = peak_idx
                while end_idx < len(flux_norm) - 1:
                    if flux_norm[end_idx] < threshold_norm:
                        break
                    end_idx += 1

                start_time_val = roi_time_mpl[start_idx]
                end_time_val = roi_time_mpl[end_idx]

            except Exception as e:
                print(f"⚠️ None branch failed: {e}")
                start_time_val = np.nan
                end_time_val = np.nan

            # --- Debug Plots ---
            if debug:
                if debug_freq_ranges is not None:
                    in_range = any((freq >= fmin and freq <= fmax) for (fmin, fmax) in debug_freq_ranges)
                    if in_range:
                        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)
                        # Left: Original flux with threshold line
                        threshold_orig = threshold_norm * peak_flux_corrected + baseline
                        axes[0].plot(np.arange(len(flux_values)), flux_values, 'bo-', label="Original Flux")
                        axes[0].axhline(threshold_orig, color='red', linestyle='--', label=f"Threshold ({fit_method})")
                        axes[0].axvline(start_idx, color='green', linestyle='--', label="Start")
                        axes[0].axvline(end_idx, color='purple', linestyle='--', label="End")
                        axes[0].set_title(f"Original Flux (Channel: {freq:.2f} MHz)")
                        axes[0].set_xlabel("Time Index")
                        axes[0].set_ylabel("Flux")
                        axes[0].legend()
                        # Right: Normalized flux with threshold line
                        axes[1].plot(np.arange(len(flux_norm)), flux_norm, 'bo-', label="Normalized Flux")
                        axes[1].axhline(threshold_norm, color='red', linestyle='--', label=f"Threshold ({fit_method})")
                        axes[1].axvline(start_idx, color='green', linestyle='--', label="Start")
                        axes[1].axvline(end_idx, color='purple', linestyle='--', label="End")
                        axes[1].set_title(f"Normalized Flux (Channel: {freq:.2f} MHz)")
                        axes[1].set_xlabel("Time Index")
                        axes[1].set_ylabel("Normalized Flux")
                        axes[1].legend()
                        plt.tight_layout()
                        plt.show()
                else:
                    # Plot for every channel
                    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), sharex=True)
                    threshold_orig = threshold_norm * peak_flux_corrected + baseline
                    axes[0].plot(np.arange(len(flux_values)), flux_values, 'bo-', label="Original Flux")
                    axes[0].axhline(threshold_orig, color='red', linestyle='--', label=f"Threshold ({fit_method})")
                    axes[0].axvline(start_idx, color='green', linestyle='--', label="Start")
                    axes[0].axvline(end_idx, color='purple', linestyle='--', label="End")
                    axes[0].set_title(f"Original Flux (Channel: {freq:.2f} MHz)")
                    axes[0].set_xlabel("Time Index")
                    axes[0].set_ylabel("Flux")
                    axes[0].legend()
                    axes[1].plot(np.arange(len(flux_norm)), flux_norm, 'bo-', label="Normalized Flux")
                    axes[1].axhline(threshold_norm, color='red', linestyle='--', label=f"Threshold ({fit_method})")
                    axes[1].axvline(start_idx, color='green', linestyle='--', label="Start")
                    axes[1].axvline(end_idx, color='purple', linestyle='--', label="End")
                    axes[1].set_title(f"Normalized Flux (Channel: {freq:.2f} MHz)")
                    axes[1].set_xlabel("Time Index")
                    axes[1].set_ylabel("Normalized Flux")
                    axes[1].legend()
                    plt.tight_layout()
                    plt.show()

            # Append results for this channel (even if nan) so that the arrays remain the same size.
            start_times.append(start_time_val)
            end_times.append(end_time_val)
            start_freqs.append(freq)
            end_freqs.append(freq)

        elif fit_mode in MODELS:
            # convert time to seconds relative to burst start
            t_sec = (roi_time_mpl - burst_start_time) * 24 * 3600
            func, pnames = MODELS[fit_mode]

            # baseline subtract
            baseline = np.min(flux_values)
            lc_corr  = flux_values - baseline
            peak_val = lc_corr[peak_idx]
            if peak_val <= 0:
                print(f"⚠️ {fit_mode} fit skipped at {freq:.2f} MHz (non-positive peak).")
                start_times.append(np.nan)
                end_times.append(np.nan)
                continue

            # initial guesses & bounds
            p0, lb, ub = [], [], []
            for pn in pnames:
                if pn == 'A':
                    p0.append(peak_val); lb.append(0);    ub.append(peak_val*10)
                elif pn in ('tau','tau1','tau2'):
                    p0.append((t_sec[-1]-t_sec[0])/5); lb.append(0);     ub.append(t_sec[-1]-t_sec[0])
                elif pn in ('sigma','sigma1','sigma2'):
                    p0.append((t_sec[-1]-t_sec[0])/4); lb.append(1e-3);  ub.append(t_sec[-1]-t_sec[0])
                elif pn in ('t0','toff'):
                    p0.append(t_sec[peak_idx]); lb.append(t_sec[0]);    ub.append(t_sec[peak_idx])
                elif pn == 'C':
                    p0.append(baseline);   lb.append(-np.inf);   ub.append(np.inf)
                else:
                    p0.append(0);          lb.append(-np.inf);   ub.append(np.inf)

            try:
                # perform the fit on the full, corrected curve
                popt, _ = optimize.curve_fit(
                    func, t_sec, lc_corr,
                    p0=p0, bounds=(lb, ub), maxfev=5000
                )
                # rebuild the full model (add baseline back)
                ymod = func(t_sec, *popt) + baseline

                # new (model‐based) version:
                C_fit  = popt[pnames.index('C')]
                S_peak = np.nanmax(ymod)

                if fit_method == "FWHM":
                    thr = C_fit + 0.5 * (S_peak - C_fit)
                else:  # "1/e"
                    thr = C_fit + (S_peak - C_fit) / np.e

                # find start: last time before peak where ymod < thr
                left_idx = np.where(t_sec <= t_sec[peak_idx])[0]
                below_left = np.where(ymod[left_idx] < thr)[0]
                if below_left.size:
                    t_start = t_sec[left_idx[below_left[-1]]]
                else:
                    t_start = np.nan

                # find end: first time after peak where ymod < thr
                right_idx = np.where(t_sec >= t_sec[peak_idx])[0]
                below_right = np.where(ymod[right_idx] < thr)[0]
                if below_right.size:
                    t_end = t_sec[right_idx[below_right[0]]]
                else:
                    t_end = np.nan

            except Exception as e:
                print(f"⚠️ {fit_mode} fit failed at {freq:.2f} MHz: {e}")
                t_start, t_end = np.nan, np.nan

            # convert back to mpl‐times
            start_times.append(burst_start_time + t_start/(24*3600))
            end_times  .append(burst_start_time + t_end  /(24*3600))
            start_freqs.append(freq)
            end_freqs.append(freq)

            # optional debug overlay
            if debug:
                plt.figure(figsize=(5,3))
                plt.plot(t_sec, flux_values, 'k-', label='data')
                plt.plot(t_sec, ymod,      'r--', label=fit_mode)
                plt.axhline(thr, color='gray', ls='--')
                plt.axvline(t_sec[peak_idx], color='gray', ls=':', label='peak')
                if not np.isnan(t_start): plt.axvline(t_start, color='green', ls='--', label='start')
                if not np.isnan(t_end):   plt.axvline(t_end,   color='blue', ls='--', label='end')
                plt.title(f"{freq:.2f} MHz ({fit_mode})")
                plt.legend(); plt.tight_layout(); plt.show()
    
    
    # ✅ Convert frequency to radial distance using density model
    e = 4.8e-10
    m_e = 9.1e-28
    pi = np.pi
    
    # ——————————————————————————————————————————————————————————————————————
    # 1) Turn all of our lists into arrays (including the freqs for start/end)
    # ——————————————————————————————————————————————————————————————————————
    peak_times   = np.array(peak_times)
    peak_freqs   = np.array(peak_freqs)
    start_times  = np.array(start_times)
    end_times    = np.array(end_times)
    start_freqs  = np.array(start_freqs)
    end_freqs    = np.array(end_freqs)

    # convert to seconds relative to burst start
    peak_t_sec   = (peak_times  - burst_start_time) * 86400
    start_t_sec  = (start_times - burst_start_time) * 86400
    end_t_sec    = (end_times   - burst_start_time) * 86400

    # alias so existing code still works
    peak_times_seconds  = peak_t_sec
    start_times_seconds = start_t_sec
    end_times_seconds   = end_t_sec

    # ——————————————————————————————————————————————————————————————————————
    # 2.1) fit on the peak curve → drift rate (linear)
    # ——————————————————————————————————————————————————————————————————————
    if len(peak_t_sec) > 1:
        slope, intercept, *_ = stats.linregress(peak_t_sec, peak_freqs)
        drift_rate = slope
    else:
        slope = intercept = drift_rate = np.nan

    print(f"   Average Drift Rate (df/dt): {drift_rate:.4f} MHz/s")

    # ───────────────────────────────────────────────────────────────────────────────
    # 2.2) Fit instantaneous drift vs. frequency → log–log line → k, α
    # df/dt = –k·f^α ⇒ f(t) = [f0^(1–α) + k(1–α)(t–t0)]^(1/(1–α))
    # ───────────────────────────────────────────────────────────────────────────────
    # backward‐difference estimate of df/dt
    t = peak_times_seconds
    f = peak_freqs

    # build and run power-law fit
    k_fit = alpha_fit = f0_fit = t0_fit = None
    if len(t)>5:
        def f_model(tt, k, alpha, f0, t0):
            dt_ = tt - t0
            if abs(alpha-1.0)<1e-3:
                return f0 * np.exp(-k*dt_)
            inner = f0**(1-alpha) + k*(1-alpha)*dt_
            return np.clip(inner,0,None)**(1/(1-alpha))

        f0_0 = float(f.max()) if f.size else 1.0
        t0_0 = float(t.min()) if t.size else 0.0
        k0   = abs(drift_rate)/max(f0_0,1e-6)
        p0   = [k0, 1.0, f0_0, t0_0]

        try:
            popt, _ = curve_fit(f_model, t, f, p0=p0, maxfev=5000)
            k_fit, alpha_fit, f0_fit, t0_fit = popt
            print(f"   Power‐law drift: df/dt = -{k_fit:.4g}·f^{alpha_fit:.2f}")
        except Exception:
            print("   Power‐law fit failed or insufficient variation.")
    else:
        print("   Insufficient points for power‐law fit")
# ──────────────────────────────────────────────────────────────

    # ——————————————————————————————————————————————————————————————————————
    # 3) Convert each of peak/start/end frequencies → radial distances
    # ——————————————————————————————————————————————————————————————————————
    # pick plasma frequency in Hz
    if emission_mechanism.upper() == "F":
        f_peak  = peak_freqs * 1e6
        f_start = start_freqs * 1e6
        f_end   = end_freqs   * 1e6
    else:  # "H"
        f_peak  = peak_freqs/2 * 1e6
        f_start = start_freqs/2 * 1e6
        f_end   = end_freqs/2   * 1e6

    density_peak  = (pi * f_peak)**2  * m_e/(4*pi*e**2)
    density_start = (pi * f_start)**2 * m_e/(4*pi*e**2)
    density_end   = (pi * f_end)**2   * m_e/(4*pi*e**2)

    # build your density→radius spline once
    r_grid = np.linspace(1.1,10,10000)
    if density_model=="saito":
        dens_grid = 1.36e6 * r_grid**(-2.14) + 1.68e8 * r_grid**(-6.13)
    elif density_model=="leblanc98":
        dens_grid = 3.3e5  * r_grid**(-2) + 4.1e6  * r_grid**(-4) + 8e7   * r_grid**(-6)
    # …elif for parkerfit, dndr_leblanc98…
    tck = splrep(dens_grid[::-1], r_grid[::-1], s=0)

    r_peak  = splev(density_peak,  tck)
    r_start = splev(density_start, tck)
    r_end   = splev(density_end,   tck)

    # ——————————————————————————————————————————————————————————————————————
    # 4) Fit velocities on each: v = dr/dt
    # ——————————————————————————————————————————————————————————————————————
    #  Peak
    mask_p = ~np.isnan(peak_t_sec) & ~np.isnan(r_peak)
    if mask_p.sum() > 1:
        vs_p, ci_p, *_ = stats.linregress(peak_t_sec[mask_p], r_peak[mask_p])
        velocity_peak = vs_p * 7e10/3e10
    else:
        velocity_peak = np.nan

    #  Start
    mask_s = ~np.isnan(start_t_sec) & ~np.isnan(r_start)
    if mask_s.sum() > 1:
        vs_s, ci_s, *_ = stats.linregress(start_t_sec[mask_s], r_start[mask_s])
        velocity_start = vs_s * 7e10/3e10
    else:
        velocity_start = np.nan

    #  End
    mask_e = ~np.isnan(end_t_sec) & ~np.isnan(r_end)
    if mask_e.sum() > 1:
        vs_e, ci_e, *_ = stats.linregress(end_t_sec[mask_e], r_end[mask_e])
        velocity_end = vs_e * 7e10/3e10
    else:
        velocity_end = np.nan

    print(f"   Electron Beam Velocity (Peak): {velocity_peak:.4f} c")
    print(f"   Electron Beam Velocity (Start): {velocity_start:.4f} c")
    print(f"   Electron Beam Velocity (End): {velocity_end:.4f} c")

    # --------------------------------------------------------------------
    # Optionally show all density‐model curves + horizontal density ranges
    # --------------------------------------------------------------------
    if show_density_models:
        # 1) Radius grid
        r_solar = np.linspace(1.1, 10.0, 5000)

        # 2) Build each model
        models = {}
        models["saito"] = 1.36e6 * r_solar**(-2.14) + 1.68e8 * r_solar**(-6.13)
        models["leblanc98"] = (
            3.3e5 * r_solar**(-2.0)
            + 4.1e6 * r_solar**(-4.0)
            + 8.0e7 * r_solar**(-6.0)
        )
        models["parkerfit"] = (
            4.8e9  / r_solar**14
            + 3e8   / r_solar**6
            + 1.39e6* r_solar**(-2.3)
            + 3e11  * np.exp(-(r_solar - 1.0) / (20.0/960.0))
        )

        # 3) Compute burst‐range densities
        #    fundamental:
        f0 = burst["start_freq"]  # MHz
        f1 = burst["end_freq"]    # MHz
        # convert MHz→Hz
        f0_Hz = f0 * 1e6
        f1_Hz = f1 * 1e6
        # plasma frequency → density
        dens0_fund = (pi * f0_Hz)**2 * m_e / (4*pi*e**2)
        dens1_fund = (pi * f1_Hz)**2 * m_e / (4*pi*e**2)
        # harmonic: f/2
        dens0_harm = (pi * (f0_Hz/2))**2 * m_e / (4*pi*e**2)
        dens1_harm = (pi * (f1_Hz/2))**2 * m_e / (4*pi*e**2)

        # 4) Plot them
        fig, ax = plt.subplots(figsize=(8,6))
        for name, dens_profile in models.items():
            ax.plot(
                r_solar, dens_profile,
                label=name, linewidth=1.5
            )

        # 5) Overlay horizontal bands
        ax.fill_between(
            [r_solar.min(), r_solar.max()],
            dens0_fund, dens1_fund,
            color='gray', alpha=0.3,
            label='Fundamental range'
        )
        ax.fill_between(
            [r_solar.min(), r_solar.max()],
            dens0_harm, dens1_harm,
            color='orange', alpha=0.3,
            label='Harmonic range'
        )

        # 6) Finalize
        ax.set_yscale('log')
        ax.set_xlabel('Radial Distance [$R_\\odot$]')
        ax.set_ylabel('Electron Density [$\\mathrm{cm}^{-3}$]')
        ax.set_title('Coronal Density Models & Burst Density Ranges')
        ax.grid(True, which='both', ls=':')
        ax.legend(loc='best', fontsize='small')
        plt.tight_layout()
        plt.show()

    # ✅ Compute beam width (Δf) for each time interval
    beam_widths = []
    beam_times = []
    normalized_beam_widths = []

    # Define time intervals for calculating Δf / f_max (e.g., 1 second steps)
    time_intervals = np.arange(peak_times_seconds.min(), peak_times_seconds.max(), 1.0)

    for t_start in time_intervals:
        t_end = t_start + 1.0

        # ✅ Find points within this time interval
        in_interval = (peak_times_seconds >= t_start) & (peak_times_seconds < t_end)
        
        if np.sum(in_interval) >= 2:
            f_start = np.min(peak_freqs[in_interval])
            f_end = np.max(peak_freqs[in_interval])
            f_max = np.mean(peak_freqs[in_interval])  # Average if multiple peaks
            
            Δf = f_end - f_start
            beam_widths.append(Δf)
            beam_times.append(t_start)

            if f_max > 0:
                normalized_beam_widths.append(Δf / f_max)
            else:
                normalized_beam_widths.append(np.nan)
                

    # ✅ Plot Results: Add 4th panel for beam width evolution
    # if we hide drift, we drop one row
    base_panels = 3 if show_drift_panel else 2
    nrows = base_panels + (1 if show_beam_width else 0)

    # height ratios: dyn‐spec, [drift], distance, [beam_width]
    heights = []
    heights.append(1)                          # panel 0: dynamic spectrum
    if show_drift_panel:
        heights.append(1)                      # panel 1: frequency drift
    heights.append(1.2)                        # panel 2 or 1: radial distance
    if show_beam_width:
        heights.append(1)                      # panel last: beam width
    fig, axs = plt.subplots(
        nrows, 1, figsize=(12, 4*nrows),
        gridspec_kw={'height_ratios': heights},
        squeeze=False
    )
    axs = axs.ravel()

    # ✅ Expand time range by 30 seconds before and after burst
    expanded_time_start = roi_time_mpl[0] - 30 / (24 * 3600)  # 30 seconds in fraction of a day
    expanded_time_end = roi_time_mpl[-1] + 30 / (24 * 3600)

    # ✅ Find indices that correspond to expanded range
    start_idx = np.searchsorted(roi_time_mpl, expanded_time_start)
    end_idx = np.searchsorted(roi_time_mpl, expanded_time_end)

    # ✅ Expand roi_data and roi_time_mpl to cover this range
    roi_time_mpl = roi_time_mpl[start_idx:end_idx]
    roi_data = roi_data[start_idx:end_idx, :]

    # use nanpercentile for robustness, and your roi_data is now 2-D
    vmin, vmax = np.nanpercentile(roi_data, [5, 95])

    if y_scale == "inverse":
        # use nanpercentile for robust color limits
        vmin, vmax = np.nanpercentile(roi_data, [5, 95])

        # build an inverse‐frequency axis (1 / MHz)
        inv_freq = 1.0 / roi_freq_values  # shape (nfreq,)

        # prepare meshgrid for pcolormesh: Time along x, inv_freq along y
        T, F = np.meshgrid(roi_time_mpl, inv_freq)

        # plot with pcolormesh so the non‐uniform y‐spacing is handled
        im = axs[0].pcolormesh(
            T, F, roi_data.T,
            shading="auto",
            cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        axs[0].invert_yaxis()

        # overlay the inverse‐frequency scatter points (peak, start, end)
        axs[0].scatter(
            peak_times,
            1.0 / peak_freqs,
            color="black", marker="+",
            label="Peak Flux", zorder=5
        )
        axs[0].scatter(
            start_times,
            1.0 / peak_freqs,
            color="red", marker="+",
            label="Start Rise", zorder=5
        )
        axs[0].scatter(
            end_times,
            1.0 / peak_freqs,
            color="blue", marker="+",
            label="End Decay", zorder=5
        )

        axs[0].set_ylabel("1 / Frequency (1/MHz)")
        axs[0].set_title(f"Dynamic Spectrum for Burst {burst['number']}")

        # add a right‐hand axis in actual MHz
        import matplotlib.ticker as ticker
        secax = axs[0].secondary_yaxis(
            'right',
            functions=(lambda inv: 1.0/inv, lambda f: 1.0/f)
        )
        secax.set_ylabel("Frequency (MHz)")
        secax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
        secax.yaxis.set_minor_formatter(ticker.NullFormatter())

        # colorbar
        fig.colorbar(im, ax=axs[0], label="Amplitude")

    elif y_scale == "log":
        extent = [roi_time_mpl[0], roi_time_mpl[-1], roi_freq_values[0], roi_freq_values[-1]]
        
        im = axs[0].imshow(roi_data.T, aspect="auto", origin="lower", extent=extent,
                        cmap="viridis", vmin=vmin, vmax=vmax)
        
        axs[0].set_yscale("log")
        
        axs[0].scatter(peak_times, peak_freqs, color="black", marker="+", label="Peak Flux", zorder=5)
        axs[0].scatter(start_times, peak_freqs, color="red", marker="+", label="Start Rise", zorder=5)
        axs[0].scatter(end_times, peak_freqs, color="blue", marker="+", label="End Decay", zorder=5)

        axs[0].set_ylabel("Frequency (MHz)")

        # (optional) mirror linear MHz on the right
        sec = axs[0].secondary_yaxis(
            'right',
            functions=(lambda f: f, lambda f: f)
        )
        sec.set_ylabel("Frequency (MHz)")

        # force the secondary axis to show more ticks and no “×10ⁿ” offset:
        sec.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
        sec.yaxis.set_major_formatter(ScalarFormatter())
        sec.yaxis.set_minor_formatter(NullFormatter())
        # hide the offset text (the “×10¹”):
        sec.yaxis.get_offset_text().set_visible(False)

    else:
        extent = [roi_time_mpl[0], roi_time_mpl[-1],
                  roi_freq_values[0], roi_freq_values[-1]]
        im = axs[0].imshow(
            roi_data.T, aspect="auto", origin="lower",
            extent=extent, cmap="viridis",
            vmin=vmin, vmax=vmax
        )
        
        axs[0].scatter(peak_times, peak_freqs, color="black", marker="+", label="Peak Flux", zorder=5)
        axs[0].scatter(start_times, peak_freqs, color="red", marker="+", label="Start Rise", zorder=5)
        axs[0].scatter(end_times, peak_freqs, color="blue", marker="+", label="End Decay", zorder=5)

        axs[0].set_ylabel("Frequency (MHz)")

    # ✅ Set x-axis format to UTC and label
    axs[0].xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    axs[0].set_xlim(expanded_time_start, expanded_time_end)
    axs[0].set_xlabel("Time (UT)")
    axs[0].set_title(f"Dynamic Spectrum for Burst {burst['number']}")
    axs[0].legend()

    # ✅ Frequency Drift Over Time (linear + non-linear model)
    # scatter the raw peak‐time vs peak‐freq

    if show_drift_panel:
        ax_drift = axs[1]
        ax_drift.scatter(t, f, color="blue", s=8, label="Peak Flux Points")
        if np.isfinite(slope):
            ax_drift.plot(t, slope*t + intercept, "r--", label="Linear Fit (Peak)")
        if k_fit is not None and alpha_fit is not None:
            t_model = np.linspace(t.min(), t.max(), 200)
            if abs(alpha_fit - 1.0) > 1e-3:
                f_curve = f_model(t_model, k_fit, alpha_fit, f0_fit, t0_fit)
            else:
                f_curve = f0_fit * np.exp(-k_fit*(t_model - t0_fit))
            ax_drift.plot(
                t_model, f_curve, "g-.",
                label=f"Power-law: df/dt={k_fit:.3g}·f^{alpha_fit:.2f}"
            )
        ax_drift.set_xlabel("Time (s since ROI start)")
        ax_drift.set_ylabel("Frequency (MHz)")
        ax_drift.set_title("Frequency Drift Over Time")
        ax_drift.legend(fontsize="small")

    # ─── Panel 2 (or 1): Radial Distance vs Time ──────────────────────────
    # If show_drift_panel=True this lives in axs[2], otherwise in axs[1]
    idx_radial = 2 if show_drift_panel else 1
    ax_radial = axs[idx_radial]

    # scatter the radial distances at peak times
    ax_radial.scatter(
        peak_t_sec, r_peak,
        color="green", marker="o", s=10,
        label="Peak r(t)"
    )
    # linear fit to the peak‐distance points
    if mask_p.sum() > 1:
        ax_radial.plot(
            peak_t_sec[mask_p],
            peak_t_sec[mask_p] * vs_p + ci_p,
            color="orange", ls="--",
            label=f"Peak v = {velocity_peak:.3f}\u00A0c"
        )

    # scatter the radial distances at start times
    ax_radial.scatter(
        start_t_sec, r_start,
        color="red", marker="x", s=10,
        label="Start r(t)"
    )
    # fit to the start‐distance points
    if mask_s.sum() > 1:
        ax_radial.plot(
            start_t_sec[mask_s],
            start_t_sec[mask_s] * vs_s + ci_s,
            color="red", ls="--",
            label=f"Start v = {velocity_start:.3f}\u00A0c"
        )

    # scatter the radial distances at end times
    ax_radial.scatter(
        end_t_sec, r_end,
        color="blue", marker="^", s=10,
        label="End r(t)"
    )
    # fit to the end‐distance points
    if mask_e.sum() > 1:
        ax_radial.plot(
            end_t_sec[mask_e],
            end_t_sec[mask_e] * vs_e + ci_e,
            color="blue", ls="--",
            label=f"End v = {velocity_end:.3f}\u00A0c"
        )

    ax_radial.set_xlabel("Time (s since ROI start)")
    ax_radial.set_ylabel(r"Radial Distance ($R_\odot$)")
    ax_radial.set_title("Electron Beam Velocity Over Time")
    ax_radial.legend(fontsize="small")


    # ─── Panel 3 (optional): Beam Width Evolution ─────────────────────────
    if show_beam_width:
        # always the last axis
        idx_beam = nrows - 1
        ax_beam = axs[idx_beam]

        ax_beam.plot(
            beam_times,
            normalized_beam_widths,
            marker="o", linestyle="-", color="purple",
            label="Beam Width / Peak Freq"
        )
        ax_beam.set_xlabel("Time (s since ROI start)")
        ax_beam.set_ylabel(r"$\Delta f / f_\mathrm{peak}$")
        ax_beam.set_title("Beam Width Evolution Over Time")
        ax_beam.legend(fontsize="small")


    plt.tight_layout()
    plt.show()