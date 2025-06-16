import os
import numpy as np
import h5py
import astropy.units as u
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter


def combine_spectra(
    date,
    output_dir,
    stokes="I",              # single string or list of strings
    rebin_dt=None,
    rebin_df=None,
    tmin=None,
    tmax=None,
    force_reload=False,
    normalization=True,
    normalization_type="median",
    normalization_threshold=0.2,
    normalization_window=5*u.min,
    exclude_freq_range=None,
    mitigate_rfi=False,
    show_dySp=False          # NEW: if True, plot dynamic spectra after combine
):
    """
    Combine low‐ and high‐band .spectra into one (or more) HDF5 files, one per requested Stokes.
    `stokes` may be e.g. "I", ["I","V"], or ["Q/I","V/I"] etc.

    If show_dySp=True, then after writing each COM file it will immediately display
    the corresponding dynamic spectrum using matplotlib.
    """

    # 1) normalize stokes parameter list
    if isinstance(stokes, str):
        stokes = [stokes]
    base = set()
    for s in stokes:
        if "/" in s:
            num, den = s.split("/")
            base.add(num)
            base.add(den)
        else:
            base.add(s)
    base = list(base)

    # build file suffix
    suffix = "_RFIremoval" if mitigate_rfi else ""
    base_tag = "_".join(base)
    low_h5  = os.path.join(output_dir, f"{date}_{base_tag}_0{suffix}.hdf5")
    high_h5 = os.path.join(output_dir, f"{date}_{base_tag}_1{suffix}.hdf5")

    # get() kwargs
    get_kwargs = {"stokes": base}
    if rebin_dt: get_kwargs["rebin_dt"] = rebin_dt
    if rebin_df: get_kwargs["rebin_df"] = rebin_df
    if tmin:     get_kwargs["tmin"]     = tmin
    if tmax:     get_kwargs["tmax"]     = tmax

    # 2) produce low & high HDF5
    for spec_idx in (0,1):
        h5file = low_h5 if spec_idx==0 else high_h5
        if force_reload and os.path.exists(h5file):
            os.remove(h5file)
        if force_reload or not os.path.exists(h5file):
            base_dir = f"/databf/nenufar-tf/LT11/{date[:4]}/{date[4:6]}/"
            sun_folder = next(
                (os.path.join(base_dir,d) for d in os.listdir(base_dir)
                 if date in d and "_SUN_TRACKING" in d), None)
            if sun_folder is None:
                raise FileNotFoundError(f"No _SUN_TRACKING for {date} under {base_dir}")
            spec_file = next(fn for fn in os.listdir(sun_folder)
                             if fn.endswith(f"_{spec_idx}.spectra"))
            spec_path = os.path.join(sun_folder, spec_file)

            print(f"📥 Writing {spec_path} → {h5file}")
            sp = Spectra(spec_path, check_missing_data=False)
            sp.pipeline.parameters["remove_channels"] = [0,1,-1]
            if mitigate_rfi:
                freq_task = TFTask.mitigate_frequency_rfi(sigma_clip=2, polynomial_degree=5)
                time_task = TFTask.mitigate_time_rfi(sigma_clip=2, polynomial_degree=5)
                sp.pipeline = TFPipeline(sp, freq_task, time_task)
            sp.get(file_name=h5file, **get_kwargs)
            del sp
        else:
            print(f"✅ Already have {h5file}")

    # helper to read each base stokes
    def _read_all(h5path):
        with h5py.File(h5path, 'r') as f:
            grp = f['SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES']
            freq = grp['frequency'][:]
            time = grp['time'][:]
            data = {b: f[f"SUB_ARRAY_POINTING_000/BEAM_000/{b}"][:] for b in base}
        return time, freq, data

    t_low,  f_low,  data_low  = _read_all(low_h5)
    t_high, f_high, data_high = _read_all(high_h5)

    # 3) align time
    dt_low  = np.median(np.diff(t_low))
    dt_high = np.median(np.diff(t_high))
    if t_low.shape==t_high.shape and np.allclose(t_low,t_high,rtol=1e-8,atol=1e-12):
        common_time = t_low
        aligned_low  = data_low
        aligned_high = data_high
    else:
        t0 = max(t_low[0], t_high[0])
        t1 = min(t_low[-1], t_high[-1])
        dt_common = min(dt_low, dt_high)
        common_time = np.arange(t0, t1+dt_common/2, dt_common)
        aligned_low  = {}
        aligned_high = {}
        for b in base:
            f_low_interp  = interp1d(t_low,  data_low[b],  axis=0,
                                     kind='linear', bounds_error=False, fill_value=np.nan)
            f_high_interp = interp1d(t_high, data_high[b], axis=0,
                                     kind='linear', bounds_error=False, fill_value=np.nan)
            aligned_low[b]  = f_low_interp(common_time)
            aligned_high[b] = f_high_interp(common_time)

    # 4) concat in freq
    combined = {}
    combined_freq = np.concatenate([f_low, f_high])
    for b in base:
        L = aligned_low[b][...,None]
        H = aligned_high[b][...,None]
        combined[b] = np.concatenate([L,H],axis=1)

    # 5) exclude frequencies
    if exclude_freq_range:
        mask = np.ones_like(combined_freq, bool)
        for mn,mx in exclude_freq_range:
            if mn is None:    mask &= combined_freq> mx
            elif mx is None:  mask &= combined_freq< mn
            else:             mask &= ~((combined_freq>=mn)&(combined_freq<=mx))
        combined_freq = combined_freq[mask]
        for b in base:
            combined[b] = combined[b][:,mask,:]

    # 6) sliding-window normalization (only on I)
    if normalization:
        wmin = normalization_window.to_value(u.min)
        print(f"🔍 Performing sliding-window normalization ({wmin:.0f} min blocks)...")
        dt_sec = np.median(np.diff(common_time))*86400.0
        block_len = max(1,int((wmin*60)/dt_sec))
        ntime,nchan,_ = combined[base[0]].shape
        mins=[]; pX=[]; mids=[]; masks=[]
        for start in range(0,ntime,block_len):
            end = min(start+block_len,ntime)
            block_data = np.stack([combined[b][start:end,:,0] for b in base],axis=0)
            mins.append(np.nanmin(block_data, axis=1))
            pX.append(np.nanpercentile(block_data, normalization_threshold*100, axis=1))
            masks.append(block_data <= pX[-1][:,None,:])
            mids.append(common_time[(start+end)//2])
        mins = np.stack(mins,0)
        pX   = np.stack(pX,0)
        mids = np.array(mids)
        masks = np.concatenate(masks,axis=1)
        safe_names = [s.replace("/","_") for s in stokes]
        resp_path = os.path.join(output_dir, f"{date}_{'_'.join(safe_names)}_responses.npz")
        np.savez(
            resp_path,
            window_times    = mids,
            resp_min        = mins,
            resp_percentX   = pX,
            freq            = combined_freq,
            background_mask = masks
        )
        print(f"   → saved response curves + mask to {resp_path}")
        if normalization_type=="median": final_resp = np.nanmedian(pX,axis=0)
        else:                       final_resp = np.nanmean(pX,axis=0)
        if "I" in base:
            i_idx = base.index("I")
            combined["I"] = combined["I"]/final_resp[i_idx][None,:,None]

    # 7) write COM files and (optionally) plot dynamic spectra
    out_paths = []
    for s in stokes:
        safe = s.replace("/","_")
        # build the array for this Stokes parameter
        if "/" in s:
            num,den = s.split("/")
            arr = combined[num] / combined[den]
        else:
            arr = combined[s]

        out_h5 = os.path.join(output_dir, f"{date}_{safe}{suffix}_COM.hdf5")
        print(f"💾 Saving combined {s} → {out_h5}")
        with h5py.File(out_h5,'w') as f:
            coords = f.create_group('SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES')
            coords.create_dataset('frequency', data=combined_freq, dtype='f8')
            coords.create_dataset('time',      data=common_time,    dtype='f8')
            f.create_dataset(f"SUB_ARRAY_POINTING_000/BEAM_000/{safe}",
                             data=arr.squeeze(), dtype='f8')
        out_paths.append(out_h5)

        # 8) dynamic‐spectrum plotting, with robust 5th/95th‐percentile color scale
    if show_dySp:
        from astropy.time import Time
        from matplotlib.dates import date2num

        for s,out_h5 in zip(stokes, out_paths):
            key = s.replace('/', '_')
            # reload data & axes from just‐written HDF5
            with h5py.File(out_h5, 'r') as f:
                grp  = f['SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES']
                freq = grp['frequency'][:]
                time_jd = grp['time'][:]          # Julian Dates
                spec = f[f"SUB_ARRAY_POINTING_000/BEAM_000/{key}"][:]

            # convert JD → matplotlib‐dates
            t_jd = Time(time_jd, format='jd', scale='utc')
            time_dt = t_jd.to_datetime()
            time_mpl = date2num(time_dt)

            # spec shape = (ntime, nchan) → transpose to (nchan, ntime)
            spec2d = spec.squeeze().T

            # compute robust color‐limits
            vmin, vmax = np.nanpercentile(spec2d, [5, 95])

            fig, ax = plt.subplots(figsize=(12,6))
            im = ax.imshow(
                spec2d,
                aspect='auto', origin='lower',
                extent=[time_mpl[0], time_mpl[-1], freq[0], freq[-1]],
                cmap='viridis', vmin=vmin, vmax=vmax
            )
            cbar = plt.colorbar(im, ax=ax, label=f"Stokes {s} amplitude")

            ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
            ax.set_xlabel('Time (UT)')
            ax.set_ylabel('Frequency (MHz)')
            ax.set_title(f"NenuFAR Stokes-{s} Dyn. Spec. {date}")
            plt.tight_layout()
            plt.show()

    return out_paths



import h5py
import numpy as np
from astropy.time import Time
from matplotlib.dates import date2num
from datetime import datetime

def load_combined_hdf5(hdf5_path, stokes=None):
    """
    Read a combined NenuFAR HDF5 (.hdf5) and return:
      - data3d:       (ntime, nchan, 1) numpy array of the desired Stokes
      - time_jd:      (ntime,) numpy array of Julian Dates
      - time_unix:    (ntime,) numpy array of UNIX timestamps (float seconds)
      - time_dt:      (ntime,) list of datetime.datetime (UTC) objects
      - time_mpl:     (ntime,) numpy array of Matplotlib datenums
      - freq:         (nchan,) numpy array of frequencies in MHz

    Parameters
    ----------
    hdf5_path : str
        Path to the combined HDF5 file.
    stokes : str or None
        Which Stokes to load, e.g. "I", "V", "V/I".  If None, the
        function will auto-detect the single dataset in the file
        aside from COORDINATES (and error if there is more than one).
    """
    with h5py.File(hdf5_path, 'r') as f:
        beam = f['SUB_ARRAY_POINTING_000/BEAM_000']
        coords = beam['COORDINATES']
        time_jd = coords['time'][:]      # (ntime,) JD
        freq    = coords['frequency'][:] # (nchan,)

        # find all data keys (exclude COORDINATES)
        candidates = [k for k in beam.keys() if k != 'COORDINATES']

        # decide which dataset to read
        if stokes is None:
            if len(candidates) != 1:
                raise ValueError(
                    f"Found multiple candidates {candidates}, please specify `stokes=`"
                )
            ds_key = candidates[0]
        else:
            # map e.g. "V/I" → "V_I"
            ds_key = stokes.replace('/', '_')
            if ds_key not in candidates:
                raise KeyError(
                    f"Requested stokes '{stokes}' → dataset '{ds_key}' not in {candidates}"
                )

        data2d = beam[ds_key][:]  # (ntime, nchan)

    # convert JD → various time formats
    t_jd      = Time(time_jd, format='jd', scale='utc')
    time_unix = t_jd.unix
    time_dt   = t_jd.to_datetime()
    time_mpl  = date2num(time_dt)

    # expand to (ntime, nchan, 1)
    data3d = data2d[..., np.newaxis]

    # diagnostics
    print(f"Loaded HDF5: {hdf5_path}")
    print(f"  Stokes dataset: {ds_key}")
    print(f"  Data shape   : {data3d.shape}  (ntime, nchan, 1)")
    print(f"  Freq range   : {freq[0]:.2f} → {freq[-1]:.2f} MHz")
    print(f"  Time range   : "
          f"{datetime.utcfromtimestamp(time_unix[0])} → "
          f"{datetime.utcfromtimestamp(time_unix[-1])}")

    return data3d, t_jd, time_unix, time_dt, time_mpl, freq


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def compare_time_axes(
    time_dt_0, time_dt_1, time_dt_com,
    downsample=20,
    zoom_start=None, zoom_end=None
):
    """
    Draw each time‐axis as an event line (|) with transparency,
    downsampling every `downsample` points. Optionally zoom to
    [zoom_start, zoom_end], which should be datetime.datetime objects.
    """
    fig, ax = plt.subplots(figsize=(12,3))

    # downsample
    t0 = time_dt_0[::downsample]
    t1 = time_dt_1[::downsample]
    tc = time_dt_com[::downsample]

    # eventplot: three rows of little ticks
    ax.eventplot([t0, t1, tc],
                 colors=['C0','C1','C2'],
                 lineoffsets=[0,1,2],
                 linelengths=0.8,
                 linewidths=1,
                 alpha=0.6)

    ax.set_yticks([0,1,2])
    ax.set_yticklabels(['Low (0)','High (1)','Combined'])
    ax.set_title("Comparison of time‐axes before & after interpolation")
    ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
    ax.set_xlabel("Time (UTC)")

    # apply zoom if requested
    if zoom_start is not None and zoom_end is not None:
        ax.set_xlim(zoom_start, zoom_end)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

def compare_time_axes_all(
    time_dt_low, time_dt_high, combined_times, 
    downsample=20, zoom_start=None, zoom_end=None
):
    """
    Compare low‐band, high‐band, and combined time axes for each Stokes.

    Parameters
    ----------
    time_dt_low : array of datetime
    time_dt_high: array of datetime
    combined_times: dict mapping stokes label → array of datetime for the combined file
    downsample: stride for plotting ticks
    zoom_start/zoom_end : optional datetime to zoom x‐axis
    """
    n = len(combined_times)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5*n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, (stoke, time_dt_com) in zip(axes, combined_times.items()):
        t0 = time_dt_low[::downsample]
        t1 = time_dt_high[::downsample]
        tc = time_dt_com[::downsample]

        ax.eventplot([t0, t1, tc],
                     colors=['C0','C1','C2'],
                     lineoffsets=[0,1,2],
                     linelengths=0.8,
                     linewidths=1,
                     alpha=0.6)

        ax.set_yticks([0,1,2])
        ax.set_yticklabels(['Low','High','Combined'])
        ax.set_title(f"Time‐axis Alignment for Stokes {stoke}")
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
        if zoom_start and zoom_end:
            ax.set_xlim(zoom_start, zoom_end)

    axes[-1].set_xlabel("Time (UTC)")
    plt.tight_layout()
    plt.show()