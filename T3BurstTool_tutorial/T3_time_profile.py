import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from matplotlib.dates import num2date

# ——————————————————————
# model definitions
# ——————————————————————
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

# ——————————————————————
# fitter
# ——————————————————————
def fit_roi_time_profiles(
    roi_data,
    roi_time,
    roi_freq,
    frequency_ranges,
    fit_method="FWHM",           # "FWHM" or "1/e"
    per_channel=False,
    show_fitplot=False,
    sigma=None,
    chi_definition='standard',
    window_start_mpl=None       # optional mpl‐date for x-axis label
):
    """
    Fit & pick best of: exponential, gaussian, biGaussian, Chrysaphi.
    Computes start/end times based on fit_method, prints rise/decay durations.
    """

    MODELS = {
        'exponential': (exponential, ['A','tau','C']),
        'gaussian':    (gaussian,    ['A','t0','tau','C']),
        'biGaussian':  (biGaussian,  ['A','t0','tau1','tau2','C']),
        'Chrysaphi':   (Chrysaphi,   ['A','tau1','tau2','toff','C'])
    }

    # build x‐axis label
    if window_start_mpl is not None:
        dt0 = num2date(window_start_mpl)
        start_str = dt0.strftime("%H:%M:%S")
        xlabel_str = f"Time (s since {start_str} UT)"
    else:
        xlabel_str = "Time (s since window start)"

    results = {}

    for fmin, fmax in frequency_ranges:
        idx = np.where((roi_freq >= fmin) & (roi_freq <= fmax))[0]
        if idx.size == 0:
            print(f"⚠️ No channels in {fmin:.2f}-{fmax:.2f} MHz, skipping.")
            continue

        if per_channel:
            channels = idx
            curves   = [roi_data[:,i].copy() for i in channels]
        else:
            channels = [idx.mean()]
            curves   = [np.nanmean(roi_data[:, idx], axis=1)]

        for chan_i, lc in zip(channels, curves):
            t       = roi_time.copy()
            C0      = float(np.nanmin(lc))
            A0      = float(np.nanmax(lc) - C0)
            t0      = float(t[np.nanargmax(lc)])
            tau0    = float((t[-1] - t[0]) / 5.0)

            peak_idx   = np.nanargmax(lc)
            t_peak     = t[peak_idx]
            flux_peak  = float(lc[peak_idx])

            key = f"{fmin:.2f}-{fmax:.2f}MHz"
            if per_channel:
                key += f"_ch{int(chan_i)}"
            results.setdefault(key, {})

            print(f"\n🏷 Band {key}:")
            print(f" → Light-curve peak: time={t_peak:.3f}s, flux={flux_peak:.3g}")

            # fit all models
            for name, (func, pnames) in MODELS.items():
                if name == 'exponential':
                    mask_valid = t >= t0
                    t_fit  = (t[mask_valid] - t0)
                    lc_fit = (lc[mask_valid] - C0)
                else:
                    mask_valid = slice(None)
                    t_fit       = t
                    lc_fit      = lc

                # build guesses & bounds
                p0 = []; lb=[]; ub=[]
                for pn in pnames:
                    if pn == 'A':
                        p0.append(A0);    lb.append(0);     ub.append(A0*10)
                    elif pn in ('tau','tau1','tau2'):
                        p0.append(tau0);  lb.append(0);     ub.append((t[-1]-t[0])*2)
                    elif pn in ('sigma','sigma1','sigma2'):
                        p0.append(tau0/2); lb.append(1e-3);  ub.append((t[-1]-t[0]))
                    elif pn in ('t0','toff'):
                        p0.append(t0);    lb.append(t[0]);  ub.append(t0)
                    elif pn == 'C':
                        p0.append(C0);    lb.append(-np.inf); ub.append(np.inf)
                    else:
                        p0.append(0);     lb.append(-np.inf); ub.append(np.inf)

                try:
                    popt, _ = curve_fit(func, t_fit, lc_fit, p0=p0, bounds=(lb,ub), maxfev=5000)

                    # full model
                    if name=='exponential':
                        ymod_full = exponential(t - t0, *popt) + C0
                        ymod_full[~mask_valid] = np.nan
                        ymod_fit = func(t_fit, *popt)
                    else:
                        ymod_full = func(t, *popt)
                        ymod_fit  = ymod_full.copy()

                    # χ²
                    dof = len(lc_fit) - len(popt)
                    sig = None
                    if sigma is not None:
                        sig = sigma[:, idx].mean(axis=1)[mask_valid]
                    chi2 = compute_chi2(lc_fit, ymod_fit, sig, chi_definition, dof)

                    # params + S_peak
                    params = {pn: float(val) for pn,val in zip(pnames,popt)}
                    S_peak = float(np.nanmax(ymod_full))
                    params['S_peak'] = S_peak

                except Exception:
                    chi2      = np.nan
                    params    = {pn: None for pn in pnames}
                    ymod_full = np.full_like(t, np.nan)

                results[key][name] = {'params':params, 'chi2':chi2, 'ymod':ymod_full}

            # pick best
            sorted_models = sorted(
                results[key].items(),
                key=lambda kv: kv[1]['chi2'] if not np.isnan(kv[1]['chi2']) else np.inf
            )
            best       = sorted_models[0][0]
            runner_up  = sorted_models[1][0] if len(sorted_models)>1 else None
            results[key]['best_model'] = best

            # pull out the records
            rec_best = results[key][best]
            ymod_best = rec_best['ymod']
            params_best = rec_best['params']
            Cb = params_best.get('C', None)
            Sp = params_best.get('S_peak', None)

            # decide thresholds
            if Cb is None or Sp is None:
                thr = None
            else:
                if fit_method.upper()=="FWHM":
                    thr = Cb + 0.5*(Sp-Cb)
                else:  # "1/e"
                    thr = Cb + (Sp-Cb)/np.e

            # locate peak index
            idx_peak = np.nanargmax(ymod_best)

            # 1) decay time always from best_model
            if thr is None:
                t_end_fit = np.nan
            else:
                # search to the right
                right = np.where(ymod_best[idx_peak:] < thr)[0]
                if right.size:
                    t_end_fit = roi_time[idx_peak + right[0]]
                else:
                    t_end_fit = np.nan

            # 2) rise time: if best_model=="exponential", fallback to runner_up
            if best=="exponential" and runner_up is not None:
                # fit the runner-up curve
                rec2 = results[key][runner_up]
                ymod2 = rec2['ymod']
                # same threshold but on ymod2:
                if thr is None:
                    t_start_fit = np.nan
                else:
                    left = np.where(ymod2[:idx_peak] < thr)[0]
                    if left.size:
                        t_start_fit = roi_time[left[-1]]
                    else:
                        # fallback to ROI start
                        t_start_fit = roi_time[0]
                rise_model = runner_up
                decay_model = best
            else:
                # both from best_model
                if thr is None:
                    t_start_fit = np.nan
                else:
                    left = np.where(ymod_best[:idx_peak] < thr)[0]
                    if left.size:
                        t_start_fit = roi_time[left[-1]]
                    else:
                        t_start_fit = roi_time[0]
                rise_model = decay_model = best

            # record in dict
            results[key]['t_start'] = float(t_start_fit)
            results[key]['t_end']   = float(t_end_fit)
            results[key]['rise_model']  = rise_model
            results[key]['decay_model'] = decay_model

            # compute and print durations
            rise_time  = t_peak - t_start_fit
            decay_time = t_end_fit - t_peak
            print(f"   → rise_time  = {rise_time:.3f}s ({rise_model} used)")
            print(f"   → decay_time = {decay_time:.3f}s ({decay_model} used)")

            # optionally plot
            if show_fitplot:
                plt.figure(figsize=(5,3))
                plt.plot(t, lc, 'k-', label='data')
                plt.axvline(t_peak, color='gray', linestyle=':', label='peak')
                for nm, mrec in results[key].items():
                    if nm in MODELS:
                        dname = nm  # keys already use 'Chrysaphi'
                        plt.plot(t, mrec['ymod'], label=dname)
                if not np.isnan(t_start_fit):
                    plt.axvline(t_start_fit, color='green', linestyle='--', label='start')
                if not np.isnan(t_end_fit):
                    plt.axvline(t_end_fit,   color='blue',  linestyle='--', label='end')
                plt.title(key)
                plt.xlabel(xlabel_str)
                plt.ylabel("Flux (arb. units)")
                plt.legend(fontsize='small')
                plt.tight_layout()
                plt.show()

            # print χ² summary
            print()
            for nm, mrec in sorted_models:
                pstr = ", ".join(f"{k}={mrec['params'][k]:.3g}" 
                                 for k in mrec['params'])
                print(f" • {nm:<12s}: {pstr}, χ²={mrec['chi2']:.3g}")
            print(f" → Best: {best}")

    return results