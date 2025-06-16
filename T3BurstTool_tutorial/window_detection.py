# window_detection.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.dates import DateFormatter, num2date


def detect_window(
    data3d,
    time_mpl,
    freq_values,
    t_start_mpl,
    t_end_mpl,
    f_min,
    f_max,
    show=True,
    outlier_sigma=None,   # drop residuals > n×σ
    y_scale="linear",     # "linear" (default) or "inverse"
    peak_drift=True       # if False, only show dyn. spec, no peaks/drift panel
):
    """
    Slice out a user‐defined time/frequency window, clean up any
    all‐NaN channels, fit a linear drift (if requested), then optionally
    show a dynamic spectrum (linear or 1/f) + overlay peaks, and (if
    peak_drift=True) show a second panel of df/dt vs time.

    Parameters
    ----------
    data3d : np.ndarray
        Full dynamic‐spectrum cube, shape (ntime, nchan, 1).
    time_mpl : np.ndarray
        Matplotlib‐float dates (days) of length ntime.
    freq_values : np.ndarray
        Frequency axis (MHz) of length nchan.
    t_start_mpl, t_end_mpl : float
        Matplotlib start/end times (in days) for the window.
    f_min, f_max : float
        Frequency bounds (MHz) for the window.
    show : bool
        If True, display plots; otherwise skip plotting.
    outlier_sigma : float or None
        If not None, perform sigma‐clipping on the peak–drift fit.
    y_scale : {"linear","inverse"}
        If "linear", plot freq on a normal vertical axis. If "inverse",
        show 1/f (MHz⁻¹) on the y‐axis (and include a right‐hand axis in MHz).
    peak_drift : bool
        If True, overlay peak markers and show the second‐panel drift fit.
        If False, only show the dynamic‐spectrum panel (no overlaid peaks,
        no drift panel).
    """

    # 1) Build 2D spec and masks
    spec2d = data3d.squeeze()                # shape (ntime, nchan)
    tmask  = (time_mpl >= t_start_mpl) & (time_mpl <= t_end_mpl)
    fmask  = (freq_values >= f_min)   & (freq_values <= f_max)

    # Extract the sub‐window
    sub   = spec2d[tmask][:, fmask]          # (ntime_sub, nchan_sub)
    t_sub = time_mpl[tmask]                  # (ntime_sub,)
    f_sub = freq_values[fmask]               # (nchan_sub,)

    # 2) Drop any channels that are entirely NaN
    valid = ~np.all(np.isnan(sub), axis=0)
    if not np.all(valid):
        dropped = np.sum(~valid)
        print(f"⚠️  Dropping {dropped} empty channels in [{f_min}–{f_max}] MHz")
    sub   = sub[:, valid]
    f_sub = f_sub[valid]

    # 3) Find the peak‐time in each remaining channel
    peak_times, peak_freqs = [], []
    for i, f in enumerate(f_sub):
        idx = np.nanargmax(sub[:, i])   # at least one non‐NaN in that column
        peak_times.append(t_sub[idx])
        peak_freqs.append(f)
    peak_times = np.array(peak_times)
    peak_freqs = np.array(peak_freqs)

    # 4) Initial linear fit of df/dt (in MHz/s)
    tsec = (peak_times - t_sub[0]) * 24 * 3600
    if len(tsec) > 1:
        slope, intercept, *_ = stats.linregress(tsec, peak_freqs)
        drift_rate = slope
    else:
        slope = intercept = drift_rate = np.nan

    # 5) Optional outlier removal by residual σ‐clipping
    if (outlier_sigma is not None) and (len(tsec) > 1):
        pred      = slope * tsec + intercept
        resid     = peak_freqs - pred
        sigma_res = np.nanstd(resid)
        keep      = np.abs(resid) <= (outlier_sigma * sigma_res)
        if np.sum(~keep):
            print(f"   ↳ removing {np.sum(~keep)} outliers (> {outlier_sigma}σ)")
        tsec, peak_freqs, peak_times = tsec[keep], peak_freqs[keep], peak_times[keep]
        if len(tsec) > 1:
            slope, intercept, *_ = stats.linregress(tsec, peak_freqs)
            drift_rate = slope

    # 6) Assemble minimal burst‐meta dict
    burst_meta = {
        "number":         1,
        "start_time":     num2date(t_start_mpl),
        "peak_time":      num2date(peak_times[peak_freqs.argmax()]) if len(peak_freqs)>0 else num2date(t_start_mpl),
        "end_time":       num2date(t_end_mpl),
        "start_freq":     f_max,
        "end_freq":       f_min,
        "start_time_mpl": t_start_mpl,
        "peak_time_mpl":  (peak_times.mean() if len(peak_times)>0 else (t_start_mpl + t_end_mpl)/2),
        "end_time_mpl":   t_end_mpl
    }

    # 7) Optional plotting
    if show:
        if peak_drift:
            # # Two‐row figure: top = dynamic spectrum, bottom = drift‐fit
            # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
            # One figure: dynamic spectrum with 
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))
        else:
            # Single‐panel: only the dynamic spectrum
            fig, ax1 = plt.subplots(1, 1, figsize=(10, 4))

        # --- TOP PANEL: Dynamic Spectrum ---

        # Determine color scale robustly (5th/95th percentile)
        vmin, vmax = np.nanpercentile(sub, [5, 95])

        if y_scale == "inverse":
            # Build an inverse‐frequency axis (1/MHz)
            inv_freq = 1.0 / f_sub         # shape (nchan_sub,)
            # Create meshgrid: time on x, inv_freq on y
            T, F = np.meshgrid(t_sub, inv_freq)

            # pcolormesh with non‐uniform y‐spacing
            im = ax1.pcolormesh(
                T, F, sub.T,
                shading="auto",
                cmap="viridis",
                vmin=vmin, vmax=vmax
            )
            ax1.invert_yaxis()  # highest freq at top of plot (smallest 1/f at top)

            # Overlay the inverse‐frequency scatter of peak times
            if peak_drift:
                ax1.scatter(
                    peak_times,
                    1.0 / peak_freqs,
                    color="red", marker="x",
                    label="I‐Peaks", zorder=5
                )

            ax1.set_ylabel("1 / Frequency (1/MHz)")
            ax1.set_title("Windowed Dyn. Spec (1/f scale)")

            # Add a right‐hand axis in linear MHz
            import matplotlib.ticker as ticker
            secax = ax1.secondary_yaxis(
                'right',
                functions=(lambda inv: 1.0/inv, lambda f: 1.0/f)
            )
            secax.set_ylabel("Frequency (MHz)")
            secax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=False))
            secax.yaxis.set_minor_formatter(ticker.NullFormatter())

            # Colorbar
            fig.colorbar(im, ax=ax1, label="Amplitude")

        else:
            # y_scale == "linear": regular imshow on freq (MHz)
            extent = [t_sub[0], t_sub[-1], f_sub[0], f_sub[-1]]
            im = ax1.imshow(
                sub.T, origin="lower", aspect="auto",
                extent=extent, cmap="viridis",
                vmin=vmin, vmax=vmax
            )

            # Overlay peaks if requested
            if peak_drift:
                ax1.scatter(peak_times, peak_freqs,
                            color="red", marker="x", label="I‐Peaks", zorder=5)

            ax1.set_ylabel("Frequency (MHz)")
            ax1.set_title("Windowed Dyn. Spec")

            # (Optional) mirror linear MHz on the right† them same as left
            # sec = ax1.secondary_yaxis(
            #     'right',
            #     functions=(lambda f: f, lambda f: f)
            # )
            # sec.set_ylabel("Frequency (MHz)")

            fig.colorbar(im, ax=ax1, label="Amplitude")

        # Format x‐axis as UTC HH:MM:SS
        ax1.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))

        # If we are not plotting drift, finish here:
        if not peak_drift:
            ax1.set_xlabel("Time (UT)")
            plt.tight_layout()
            plt.show()
            return {
                "burst":           burst_meta,
                "full_data":       data3d,
                "full_time_mpl":   time_mpl,
                "roi_data":        sub[:, :, np.newaxis],
                "roi_time_mpl":    t_sub,
                "roi_freq_values": f_sub,
                "peak_times":      peak_times,
                "peak_freqs":      peak_freqs,
                "drift_rate":      drift_rate,
                "drift_intercept": intercept
            }

        # If we reach here, we still want to draw the lower “drift‐fit” panel:

        # # --- BOTTOM PANEL: Frequency Drift Plot ---
        # ax2.scatter(tsec, peak_freqs, c="blue", label="Peak Times", s=20)
        # if not np.isnan(drift_rate):
        #     ax2.plot(tsec, slope*tsec + intercept, "r--",
        #              label=f"df/dt = {drift_rate:.3f} MHz/s")
        # ax2.set_xlabel("Seconds since window start")
        # ax2.set_ylabel("Frequency (MHz)")
        # ax2.set_title("Frequency Drift Over Time")
        # ax2.legend()

        # plt.tight_layout()
        # plt.show()

    # 8) Return the dictionary exactly as analyze_burst_velocity expects
    return {
        "burst":           burst_meta,
        "full_data":       data3d,
        "full_time_mpl":   time_mpl,
        "roi_data":        sub[:, :, np.newaxis],
        "roi_time_mpl":    t_sub,
        "roi_freq_values": f_sub,
        "peak_times":      peak_times,
        "peak_freqs":      peak_freqs,
        "drift_rate":      drift_rate,
        "drift_intercept": intercept
    }

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def stokes_peak(
    roi_data, 
    roi_time_mpl, 
    roi_freq_values, 
    peak_times, 
    peak_freqs, 
    title=None
):
    """
    Overlay “I‐peak” points onto a Stokes‐spectrum, 
    but restricted to the SAME ROI that was used in detect_window().

    Parameters
    ----------
    roi_data : 3D numpy array, shape (ntime_roi, nchan_roi, 1)
        This should be exactly the `roi_data` you got out of detect_window():
            roi_data = sub[:, :, np.newaxis]
        where `sub` was the windowed 2D slice.

    roi_time_mpl : 1D numpy array of length ntime_roi
        Matplotlib datenums (days) for each time sample within the ROI.

    roi_freq_values : 1D numpy array of length nchan_roi
        Frequencies (MHz) corresponding to each column in the ROI.

    peak_times : 1D numpy array
        Matplotlib datenums (days) of each “I‐peak” within your ROI 
        (as returned by detect_window).

    peak_freqs : 1D numpy array
        Frequencies (MHz) of each “I‐peak” (same length as peak_times).

    title : str or None
        If provided, will be used as the plot title.

    Notes
    -----
    - This routine assumes that `roi_data` was produced by:
          sub = spec2d_full[tmask][:, fmask]
          roi_data = sub[..., np.newaxis]
      and that `roi_time_mpl = t_sub` and `roi_freq_values = f_sub`.

    - If you want to overlay “I‐peak” points onto your Stokes‐V
      or Stokes‐V/I spectrum, simply pass the already‐windowed
      `roi_data_V <-- data3d_V_COM[tmask][:, fmask, :]`.
      For example, after detect_window you have:
          results = detect_window(...)
          sub_I  = results["roi_data"]         # shape = (ntime_roi, nchan_roi, 1)
          t_sub  = results["roi_time_mpl"]     # (ntime_roi,)
          f_sub  = results["roi_freq_values"]  # (nchan_roi,)
          peaks  = results["peak_times"], results["peak_freqs"]

      Now you build the corresponding Stokes‐V slice:
          # full V data3d: data3d_V_COM  with shape (ntime_full, nchan_full, 1)
          mask_t = (time_mpl_V_COM >= t_sub[0]) & (time_mpl_V_COM <= t_sub[-1])
          mask_f = (freq_V_COM    >= f_sub[0]) & (freq_V_COM    <= f_sub[-1])
          sub_V  = data3d_V_COM[mask_t][:, mask_f, 0]    # (ntime_roi, nchan_roi)

      Then call:
          stokes_peak(
              roi_data   = sub_V[..., np.newaxis],
              roi_time_mpl   = t_sub,
              roi_freq_values = f_sub,
              peak_times  = peak_times,
              peak_freqs  = peak_freqs,
              title       = "Stokes‐V (windowed) with I‐peaks"
          )
    """

    # squeeze off that last singleton axis, so we get a 2D array:
    spec2d = roi_data.squeeze()   # now shape = (ntime_roi, nchan_roi)

    ntime_roi, nchan_roi = spec2d.shape
    assert len(roi_time_mpl) == ntime_roi, \
        f"roi_time_mpl length {len(roi_time_mpl)} != spec2d.shape[0] {ntime_roi}"
    assert len(roi_freq_values) == nchan_roi, \
        f"roi_freq_values length {len(roi_freq_values)} != spec2d.shape[1] {nchan_roi}"
    assert len(peak_times) == len(peak_freqs), \
        "peak_times and peak_freqs must have same length"

    # Build the “extent” for imshow: [t0, t1, f0, f1]
    extent = [
        roi_time_mpl[0],
        roi_time_mpl[-1],
        roi_freq_values[0],
        roi_freq_values[-1],
    ]

    plt.figure(figsize=(10, 5))
    pcm = plt.imshow(
        spec2d.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis"
    )
    cbar = plt.colorbar(pcm, label="Amplitude")

    # Only plot those peaks that actually fall within this ROI window:
    mask = (peak_times >= roi_time_mpl[0]) & (peak_times <= roi_time_mpl[-1]) & \
           (peak_freqs >= roi_freq_values[0]) & (peak_freqs <= roi_freq_values[-1])

    plt.scatter(
        peak_times[mask],
        peak_freqs[mask],
        c="r",
        marker="x",
        s=30,
        label="I‐peaks"
    )

    if title:
        plt.title(title)
    plt.xlabel("Time (UT)")
    plt.ylabel("Frequency (MHz)")
    plt.gca().xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()