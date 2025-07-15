import numpy as np
import os
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from obspy import Stream, Trace
from obspy.signal.polarization import polarization_analysis
from obspy.signal.filter import bandpass
from obspy import UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees, gps2dist_azimuth
from obspy.taup import TauPyModel
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import hilbert
from scipy.optimize import minimize
import emcee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pygmt
import warnings
from pyproj import Geod

warnings.simplefilter("ignore")


def min_max_dist_to_grid(ev_lat, ev_lon):
    src_lons = np.full(len(grid_coords), ev_lon)
    src_lats = np.full(len(grid_coords), ev_lat)
    az12, az21, dist_m = geod.inv(src_lons, src_lats, grid_coords[:, 1], grid_coords[:, 0])
    dist_deg = dist_m / 111195

    idx_min = np.argmin(dist_deg)
    idx_max = np.argmax(dist_deg)

    return (
        dist_deg[idx_min],
        dist_deg[idx_max],
        grid_coords[idx_min][0], grid_coords[idx_min][1],  # lat/lon of closest
        grid_coords[idx_max][0], grid_coords[idx_max][1],  # lat/lon of furthest
    )

# ========================================================================== 80
def polarization_analysis_plot(st, win_len=1.0, win_frac=0.05,
                               freqmin=0.5, freqmax=10.0, plot_title="Polarization Analysis"):
    if len(st) != 3:
        raise ValueError("Stream must have 3 components (Z, N/1, E/2).")

    # Align time windows
    start = max(tr.stats.starttime for tr in st)
    end = min(tr.stats.endtime for tr in st)
    st = st.copy().trim(starttime=start, endtime=end, pad=True, fill_value=0)
    st.detrend("linear").taper(0.05).filter("bandpass", freqmin=freqmin, freqmax=freqmax)

    # Identify components
    comp_map = {'Z': None, 'H1': None, 'H2': None}
    for tr in st:
        ch = tr.stats.channel[-1].upper()
        if ch == 'Z':
            comp_map['Z'] = tr
        elif ch in ['N', '1']:
            comp_map['H1'] = tr
        elif ch in ['E', '2']:
            comp_map['H2'] = tr

    if None in comp_map.values():
        raise ValueError("Missing one of: Z, N/1, E/2")

    # Assign and relabel traces
    trZ = comp_map['Z'].copy()
    trN = comp_map['H1'].copy()
    trE = comp_map['H2'].copy()
    trZ.stats.channel = "BHZ"
    trN.stats.channel = "BHN"
    trE.stats.channel = "BHE"

    st_fixed = Stream([trZ, trN, trE])
    times = trZ.times(reftime=trZ.stats.starttime)

    # Polarization analysis
    paz = polarization_analysis(
        stream=st_fixed,
        win_len=win_len,
        win_frac=win_frac,
        frqlow=freqmin,
        frqhigh=freqmax,
        stime=start,
        etime=end,
        var_noise=1e-10,
        method="pm",
        verbose=False
    )
    print(paz)
    if 'rectilinearity' not in paz:
        print("polarization_analysis returned keys:", paz.keys())
        raise RuntimeError("polarization_analysis did not return 'rectilinearity'.")

    times_paz = paz['timestamp']
    rect = paz['rectilinearity']
    incidence = paz['incidence']

    rect_thresh = 0.7
    is_p_wave = (rect > rect_thresh) & (np.abs(incidence - 0) < 25)
    is_s_wave = (rect > rect_thresh) & (np.abs(incidence - 90) < 25)

    # Plotting
    fig, ax = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    ax[0].plot(times, trZ.data, label="Z")
    ax[0].plot(times, trN.data, label="N/1")
    ax[0].plot(times, trE.data, label="E/2")
    ax[0].legend()
    ax[0].set_ylabel("Amplitude")
    ax[0].set_titl_
    
# ========================================================================== 80    

def compute_polarization_manual(st, win_len=1.0, win_step=0.05):
    """
    Perform polarization analysis via eigen decomposition of 3-component data.
    Returns:
        - times (in seconds from trace start)
        - rectilinearity
        - incidence angle
    """
    # Identify components
    comp_map = {'Z': None, 'H1': None, 'H2': None}
    for tr in st:
        ch = tr.stats.channel[-1].upper()
        if ch == 'Z':
            comp_map['Z'] = tr
        elif ch in ['N', '1']:
            comp_map['H1'] = tr
        elif ch in ['E', '2']:
            comp_map['H2'] = tr
    if None in comp_map.values():
        raise ValueError("Input stream must contain Z, N/1, and E/2 components")

    trZ, trN, trE = comp_map['Z'].copy(), comp_map['H1'].copy(), comp_map['H2'].copy()

    # Align and trim all traces
    start = max(tr.stats.starttime for tr in [trZ, trN, trE])
    end = min(tr.stats.endtime for tr in [trZ, trN, trE])
    for tr in [trZ, trN, trE]:
        tr.trim(starttime=start, endtime=end)

    sr = trZ.stats.sampling_rate
    npts = trZ.stats.npts
    data = np.vstack([trZ.data, trN.data, trE.data]).T  # shape (npts, 3)

    win_samples = int(win_len * sr)
    step_samples = int(win_step * sr)

    times = []
    rectilinearity = []
    incidence = []

    for i in range(0, npts - win_samples, step_samples):
        segment = data[i:i+win_samples]
        segment -= segment.mean(axis=0)

        cov = segment.T @ segment
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]

        l1, l2, l3 = eigvals[order]
        v1 = eigvecs[:, order[0]]

        rect = 1 - (l2 + l3) / l1
        inc = np.arccos(np.abs(v1[0])) * 180 / np.pi

        rectilinearity.append(rect)
        incidence.append(inc)

        # Relative time (seconds from start)
        t_mid = (i + win_samples // 2) / sr
        times.append(t_mid)

    return np.array(times), np.array(rectilinearity), np.array(incidence)

# ========================================================================== 80    
def plot_polarization_result(
    trZ,
    times_paz,
    rect,
    incidence,
    trN=None,
    trE=None,
    rect_thresh=0.7,
    p_incidence_max=30,
    s_incidence_min=45,
    amp_thresh=0.01,
    p_pick_search_start=0.0,
    p_pick_search_end=None,
    filename=None,
    show_plot=True,
    true_p_arrival=None,
    true_s_arrival=None
):
    sr = trZ.stats.sampling_rate
    times = trZ.times(reftime=trZ.stats.starttime)
    amp_log = np.log10(np.max(np.abs(trZ.data)))

    rect = np.asarray(rect)
    incidence = np.asarray(incidence)
    data_z = np.asarray(trZ.data)
    data_amp = np.abs(data_z[(times_paz * sr).astype(int)])

    p_mask = (
        (rect > rect_thresh)
        & (incidence < p_incidence_max)
        & (data_amp > amp_thresh * np.max(np.abs(data_z)))
    )

    p_times = times_paz[p_mask]
    valid_p_times = p_times[(p_times > p_pick_search_start)]
    if p_pick_search_end is not None:
        valid_p_times = valid_p_times[valid_p_times < p_pick_search_end]
    p_pick = valid_p_times[0] if len(valid_p_times) > 0 else None

    s_mask = (
        (rect > rect_thresh)
        & (incidence > s_incidence_min)
        & (data_amp > amp_thresh * np.max(np.abs(data_z)))
    )

    s_times = times_paz[s_mask]
    if p_pick:
        s_times = s_times[s_times > p_pick]
    s_pick = s_times[0] if len(s_times) > 0 else None

    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    axs[0].plot(times, trZ.data, label="Z", color="black")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    axs[1].plot(times_paz, incidence, label="Incidence Angle", color="steelblue")
    axs[1].set_ylabel("Incidence (°)")
    axs[1].legend()

    axs[2].plot(times_paz, rect, label="Rectilinearity", color="green")
    axs[2].axhline(rect_thresh, linestyle="--", color="gray")
    axs[2].set_ylabel("Rect.")
    axs[2].legend()

    axs[3].plot(times, trZ.data, label="Z", color="black")
    axs[3].scatter(p_times, np.interp(p_times, times, trZ.data), color="blue", s=8, label="P-polarized")
    axs[3].scatter(s_times, np.interp(s_times, times, trZ.data), color="red", s=8, label="S-polarized")
    axs[3].set_ylabel("Z with P/S")
    axs[3].legend()

    if trN is not None and trE is not None:
        energy_p = np.zeros_like(trZ.data)
        energy_s = np.zeros_like(trZ.data)
        for pt in p_times:
            idx = int(pt * sr)
            if 0 <= idx < len(energy_p):
                energy_p[idx] = trZ.data[idx] ** 2
        for st in s_times:
            idx = int(st * sr)
            if 0 <= idx < len(energy_s):
                energy_s[idx] = trN.data[idx] ** 2 + trE.data[idx] ** 2

        axs[4].plot(times, trZ.data, color="black", label="Z", alpha=0.4)
        axs[4].plot(times, trN.data, color="darkgreen", label="N", alpha=0.4)
        axs[4].plot(times, trE.data, color="orange", label="E", alpha=0.4)
        axs[4].plot(times, energy_s, color="red", label="S Energy (N+E)", linewidth=2)        
        axs[4].plot(times, energy_p, color="blue", label="P Energy (Z)", linewidth=2)

        axs[4].set_ylabel("Energy / Amplitude")
        axs[4].legend()
    else:
        axs[4].axis("off")

    axs[-1].set_xlabel("Time (s)")

    title = f"Vertical Component with Polarization Analysis (log$_{{10}}$ Amplitude ≈ {amp_log:.2f})"
    if p_pick:
        title += f"\nEstimated P Arrival ≈ {p_pick:.2f} s"
    if s_pick:
        title += f", S Arrival ≈ {s_pick:.2f} s"
    fig.suptitle(title)

    if p_pick:
        for ax in axs:
            ax.axvline(p_pick, color="blue", linestyle="--")
    if s_pick:
        for ax in axs:
            ax.axvline(s_pick, color="red", linestyle="--")

    if true_p_arrival:
        p_true_sec = (true_p_arrival - trZ.stats.starttime)
        for ax in axs:
            ax.axvline(p_true_sec, color="blue", linestyle=":")

    if true_s_arrival:
        s_true_sec = (true_s_arrival - trZ.stats.starttime)
        for ax in axs:
            ax.axvline(s_true_sec, color="red", linestyle=":")

    fig.tight_layout()
    if filename:
        fig.savefig(filename)
    if show_plot:
        plt.show()
    else:
        plt.close()

    return p_pick, s_pick



# ========================================================================== 80
# === Configuration ===
np.random.seed(42)  # For reproducibility

NETWORK = "IU"
STATION = "ANMO"
CHANNEL = "BH?"
LOCATION = "00"
DATE = "2019-07-06"
MIN_MAG = 5.0
LAT_RANGE = (32.0, 40.0)
LON_RANGE = (-110.0, -100.0)
Z_RANGE   = (0.0, 3.0)

# Randomized initial guess within bounds
LAT_CEN = np.random.uniform(*LAT_RANGE)
LON_CEN = np.random.uniform(*LON_RANGE)
Z_CEN   = np.random.uniform(*Z_RANGE)

MAXRADIUS = 80
MODEL = "ak135"
INVERSION_METHOD = "gradient"  # or "grid+mcmc"
num_iter = 10
N_WORKERS = 8
WAVEFORM_FILE = "cached_waveform.mseed"
RESPONSE_FILE = "cached_response.xml"
os.makedirs("picks", exist_ok=True)
os.makedirs("picks/waveform_plots", exist_ok=True)
os.makedirs("picks/waveforms", exist_ok=True)
os.makedirs("picks/polarization", exist_ok=True)

# Coarse grid resolution
N_LAT_COURSE = 6
N_LON_COURSE = 6
N_Z_COURSE = 4

# === Arrival Picker Configuration ===
PICK_METHOD = "svd"
PICK_METHODS = ["hilbert", "stalta", "random", "svd"]
if PICK_METHOD not in PICK_METHODS:
    raise ValueError(f"Invalid PICK_METHOD. Choose from {PICK_METHODS}")
else:
    print(f"Using PICK_METHOD: {PICK_METHOD}")

FILTER = (0.5, 5.0)
STA = 1.0
LTA = 10.0
TRIGGER_ON = 3.0
TRIGGER_OFF = 0.5
pick_before = 0
pick_after = 0
filter_buffer = 2.0

# === Load waveform and response ===
t0 = UTCDateTime(DATE)
t1 = t0 + 86400

def waveform_is_valid(filename, network, station, location, channel, starttime, endtime, min_bytes=1_000_000):
    try:
        if not os.path.exists(filename) or os.path.getsize(filename) < min_bytes:
            print("[Info] Waveform file missing or too small.")
            return False
        st = read(filename)
        valid_traces = [
            tr for tr in st
            if tr.stats.network == network and
               tr.stats.station == station and
               tr.stats.location.strip() == location.strip()
        ]
        if len(valid_traces) == 0:
            print("[Warning] No matching traces found in waveform file.")
            return False

        channels_present = set(tr.stats.channel[-1] for tr in valid_traces)
        # Check for Z explicitly
        has_Z = 'Z' in channels_present

        # Allow N or 1 for North component
        has_N = 'N' in channels_present or '1' in channels_present

        # Allow E or 2 for East component
        has_E = 'E' in channels_present or '2' in channels_present

        # Determine missing components
        missing = set()
        if not has_Z:
            missing.add('Z')
        if not has_N:
            missing.add('N/1')
        if not has_E:
            missing.add('E/2')

        if missing:
            print(f"Missing components: {missing}")

        if len(missing) == 3:
            print("[Warning] None of Z/N/E components found.")
            return False
        elif missing:
            print(f"[Warning] Missing components: {missing}. Proceeding with partial data.")

        times_ok = all(
            abs(tr.stats.starttime - starttime) < 10 and abs(tr.stats.endtime - endtime) < 10
            for tr in valid_traces
        )
        if not times_ok:
            print("[Warning] Time coverage not valid for all traces.")
            return False

        return True
    except Exception as e:
        print(f"[Error] Failed to validate waveform file: {e}")
        return False


client = Client("IRIS")
if waveform_is_valid(WAVEFORM_FILE, NETWORK, STATION, LOCATION, CHANNEL, t0, t1):
    print("Loading waveform from cache...")
    st = read(WAVEFORM_FILE)
else:
    print("Downloading waveform...")
    st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNEL, t0, t1)
    st.write(WAVEFORM_FILE, format="MSEED")

print(f"Waveform contains {len(st)} traces.")

if os.path.exists(RESPONSE_FILE):
    inv = read_inventory(RESPONSE_FILE)
else:
    inv = client.get_stations(network=NETWORK, station=STATION, location=LOCATION,
                              channel=CHANNEL, level="response", starttime=t0)
    inv.write(RESPONSE_FILE, format="STATIONXML")

# ========================================================================== 80
# === Phase picking ===
model = TauPyModel(MODEL)
cat = client.get_events(starttime=t0, endtime=t1, minmagnitude=MIN_MAG,
                        latitude=LAT_CEN, longitude=LON_CEN, maxradius=MAXRADIUS)

# Extract metadata arrays from the catalog
origins = [ev.preferred_origin() or ev.origins[0] for ev in cat]
magnitudes = [ev.preferred_magnitude() or ev.magnitudes[0] for ev in cat]

event_lats = np.array([o.latitude for o in origins])
event_lons = np.array([o.longitude for o in origins])
event_depths = np.array([o.depth / 1000 for o in origins])  # km
origin_times = np.array([o.time for o in origins])
event_mags = np.array([m.mag for m in magnitudes])

observed_arrivals = np.full(len(cat), np.nan)


# ========================================================================== 80
# Search grid for max/min distance from event

minlat, maxlat = LAT_RANGE
minlon, maxlon = LON_RANGE

# Grid resolution, degrees
dlat = 0.5
dlon = 0.5

# Generate grid of lat/lon points
lats = np.arange(minlat, maxlat + dlat, dlat)
lons = np.arange(minlon, maxlon + dlon, dlon)
lon_grid, lat_grid = np.meshgrid(lons, lats)

# Vectorize grid point coordinates
grid_lats, grid_lons = np.meshgrid(lat_grid, lon_grid, indexing="ij")
grid_lats_flat = grid_lats.ravel()
grid_lons_flat = grid_lons.ravel()

# Vectorized computation of distance from each event to all grid points
event_coords = np.vstack([event_lats, event_lons]).T
grid_coords = np.vstack([grid_lats_flat, grid_lons_flat]).T

# Compute closest and furthest distances (degrees) for each event

# Flattened grid points (precomputed)
grid_coords = np.column_stack((grid_lats.ravel(), grid_lons.ravel()))

# Pre-create Geod instance (WGS84)
geod = Geod(ellps="WGS84")

dist_deg_close = np.zeros(len(event_lats))
dist_deg_far = np.zeros(len(event_lats))
lat_close = np.zeros(len(event_lats))
lon_close = np.zeros(len(event_lats))
lat_far = np.zeros(len(event_lats))
lon_far = np.zeros(len(event_lats))

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = {
        executor.submit(min_max_dist_to_grid, ev_lat, ev_lon): idx
        for idx, (ev_lat, ev_lon) in enumerate(zip(event_lats, event_lons))
    }

    for future in tqdm(as_completed(futures), total=len(futures), desc="Distance computation"):
        idx = futures[future]
        try:
            dmin, dmax, lat_min, lon_min, lat_max, lon_max = future.result()
            dist_deg_close[idx] = dmin
            dist_deg_far[idx] = dmax
            lat_close[idx] = lat_min
            lon_close[idx] = lon_min
            lat_far[idx] = lat_max
            lon_far[idx] = lon_max
        except Exception as e:
            print(f"[!] Failed for event {idx}: {e}")
            dist_deg_close[idx] = np.nan
            dist_deg_far[idx] = np.nan
            lat_close[idx] = lon_close[idx] = np.nan
            lat_far[idx] = lon_far[idx] = np.nan
 

# Actual arrivals (for testing)

sta_lon = inv[0][0].longitude
sta_lat = inv[0][0].latitude

# Broadcast station location and compute distances
_, _, dists_m = geod.inv(
    np.full_like(event_lons, sta_lon),
    np.full_like(event_lats, sta_lat),
    event_lons,
    event_lats,
)
# Convert meters to degrees (approximate: 1 deg = 111.195 km)
distances_deg = dists_m / 111195.0

from concurrent.futures import ProcessPoolExecutor

def get_arrival_time(model_name, distance, depth, phase):
    """
    Computes first arrival time for a given (distance, depth).
    """
    model = TauPyModel(model_name)
    arrivals = model.get_travel_times(
        source_depth_in_km=depth,
        distance_in_degree=distance,
        phase_list=[phase]
    )
    return arrivals[0].time if arrivals else np.nan

def compute_arrival_times_parallel_futures(model_name, distances_deg, depths_km, phase="P", max_workers=None):
    """
    Computes arrival times in parallel using concurrent.futures.
    
    Parameters:
    - model_name : str : TauP model (e.g., 'iasp91')
    - distances_deg : np.ndarray : epicentral distances in degrees
    - depths_km : np.ndarray : source depths in km
    - phase : str : seismic phase to calculate ('P', 'S', etc.)
    - max_workers : int or None : number of parallel workers
    
    Returns:
    - np.ndarray of arrival times (seconds)
    """
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(get_arrival_time, model_name, dist, depth, phase)
            for dist, depth in zip(distances_deg, depths_km)
        ]
        arrival_times = [f.result() for f in futures]

    return np.array(arrival_times)


P_arrival_times = origin_times + compute_arrival_times_parallel_futures(MODEL, distances_deg, event_depths, phase="P") 
S_arrival_times = origin_times + compute_arrival_times_parallel_futures(MODEL, distances_deg, event_depths, phase="S") 

# arrivals = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg, phase_list=["P"])

# ========================================================================== 80
# Cut individual segments for potential picking

# for evt_idx in range(len(cat)):
# # try:
#     dist_deg_close_event = dist_deg_close[evt_idx]
#     dist_deg_far_event = dist_deg_far[evt_idx]
#     arrivals_close = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg_close[evt_idx], phase_list=["P"])
#     arrivals_close_P = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg_close[evt_idx], phase_list=["P"])
#     arrivals_close_S = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg_close[evt_idx], phase_list=["S"])

#     arrivals_far = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg_far[evt_idx], phase_list=["S"])
#     arrivals_far_P = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg_far[evt_idx], phase_list=["P"])
#     arrivals_far_S = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg_far[evt_idx], phase_list=["S"])

#     if not arrivals_close:
#         # print(f"[Warning] No arrivals close found for {evt_idx}")
#         # return None
#         t_pred_start = origin_times[evt_idx]
#     elif arrivals_close:
#         t_pred_start = origin_times[evt_idx] + arrivals_close[0].time

#     if not arrivals_far:
#         print(f"[Warning] No arrivals far found for {evt_idx}")
#         # return None

#     t_pred_end  = origin_times[evt_idx] + arrivals_far[0].time        
        
#     st_win = st.copy().trim(starttime=t_pred_start - pick_before, endtime=t_pred_end + pick_after)
#     st_win.filter("bandpass", freqmin=FILTER[0], freqmax=FILTER[1], corners=4, zerophase=True)
#     st_win = st_win.trim(starttime=t_pred_start - pick_before + filter_buffer, endtime=t_pred_end + pick_after)

#     st_win.write(f"picks/waveforms/event_{evt_idx:03d}.mseed", format="MSEED")

#     # ========================================================================== 80
#     # Run polarization analysis
#     times_paz, rect, incidence = compute_polarization_manual(st_win)

#     # Grab the Z trace
#     trZ = st_win.select(component="Z")[0]
#     trN = st_win.select(component="1")[0]
#     trE = st_win.select(component="2")[0]


#     # Run the plotting function
#     p_pick, s_pick = plot_polarization_result(
#         trZ=trZ,
#         trN=trN,
#         trE=trE,
#         times_paz=times_paz,
#         rect=rect,
#         incidence=incidence,
#         rect_thresh=0.7,
#         p_incidence_max=30,
#         s_incidence_min=45,
#         amp_thresh=0.01,
#         p_pick_search_start=5.0,
#         filename=f"picks/polarization/polarization_output_event_{evt_idx:03d}.png",
#         show_plot=False,
#         true_p_arrival=P_arrival_times[evt_idx],
#         true_s_arrival=S_arrival_times[evt_idx]
#     )

def process_segment_and_polarization(evt_idx):
    try:
        dist_deg_close_event = dist_deg_close[evt_idx]
        dist_deg_far_event = dist_deg_far[evt_idx]

        arrivals_close = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg_close_event, phase_list=["P"])
        arrivals_far = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg_far_event, phase_list=["S"])

        if not arrivals_close:
            t_pred_start = origin_times[evt_idx]
        else:
            t_pred_start = origin_times[evt_idx] + arrivals_close[0].time

        if not arrivals_far:
            print(f"[Warning] No arrivals far found for event {evt_idx}")
            return

        t_pred_end = origin_times[evt_idx] + arrivals_far[0].time

        st_win = st.copy().trim(starttime=t_pred_start - pick_before, endtime=t_pred_end + pick_after)
        st_win.filter("bandpass", freqmin=FILTER[0], freqmax=FILTER[1], corners=4, zerophase=True)
        st_win.trim(starttime=t_pred_start - pick_before + filter_buffer, endtime=t_pred_end + pick_after)

        st_win.write(f"picks/waveforms/event_{evt_idx:03d}.mseed", format="MSEED")

        times_paz, rect, incidence = compute_polarization_manual(st_win)

        trZ = st_win.select(component="Z")[0]
        trN = st_win.select(component="1")[0]
        trE = st_win.select(component="2")[0]

        p_pick, s_pick = plot_polarization_result(
            trZ=trZ,
            trN=trN,
            trE=trE,
            times_paz=times_paz,
            rect=rect,
            incidence=incidence,
            rect_thresh=0.7,
            p_incidence_max=30,
            s_incidence_min=45,
            amp_thresh=0.01,
            p_pick_search_start=5.0,
            filename=f"picks/polarization/polarization_output_event_{evt_idx:03d}.png",
            show_plot=False,
            true_p_arrival=P_arrival_times[evt_idx],
            true_s_arrival=S_arrival_times[evt_idx]
        )
    except Exception as e:
        print(f"[Error] Failed processing event {evt_idx}: {e}")


from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = [executor.submit(process_segment_and_polarization, evt_idx) for evt_idx in range(len(cat))]
    for f in tqdm(as_completed(futures), total=len(cat), desc="Processing segments"):
        _ = f.result()  # Optional: collect result if you modify function to return something


# ========================================================================== 80
true_lon = inv[0][0].longitude
true_lat = inv[0][0].latitude
true_elv = inv[0][0].elevation / 1000  # Convert


# === GMT Mercator Map ===
print("Plotting GMT Mercator map...")
lat_min_data = min(event_lats ) - 2
lat_max_data = max(event_lats ) + 2
lon_min_data = min(event_lons ) - 2
lon_max_data = max(event_lons ) + 2

lat_min_input, lat_max_input = LAT_RANGE
lon_min_input, lon_max_input = LON_RANGE

lat_min = min(lat_min_data, lat_min_input)
lat_max = max(lat_max_data, lat_max_input)
lon_min = min(lon_min_data, lon_min_input)
lon_max = max(lon_max_data, lon_max_input)

region = [lon_min, lon_max, lat_min, lat_max]


fig = pygmt.Figure()
fig.basemap(region=region, projection="M8i", frame=["af", f"+tStation Inversion: {STATION}"])
fig.grdimage(grid="@earth_relief_01m", region=region, projection="M8i", shading=True)

# Plot translucent grid box showing inversion search area
fig.plot(
    x=[LON_RANGE[0], LON_RANGE[1], LON_RANGE[1], LON_RANGE[0], LON_RANGE[0]],
    y=[LAT_RANGE[0], LAT_RANGE[0], LAT_RANGE[1], LAT_RANGE[1], LAT_RANGE[0]],
    pen="1p,gray",
    transparency=70,
    fill="white"
)

# Draw grid lines inside search area
lon_ticks = np.arange(LON_RANGE[0], LON_RANGE[1] + 1e-5, 1.0)
lat_ticks = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 1e-5, 1.0)
for lon in lon_ticks:
    fig.plot(x=[lon, lon], y=[LAT_RANGE[0], LAT_RANGE[1]], pen="0.25p,gray,-")
for lat in lat_ticks:
    fig.plot(x=[LON_RANGE[0], LON_RANGE[1]], y=[lat, lat], pen="0.25p,gray,-")

# Plot events and stations
fig.plot(x=event_lons, y=event_lats, style="c0.15c", fill="blue", pen="black", label="Events")
fig.plot(x=[true_lon], y=[true_lat], style="a0.3c", fill="green", pen="black", label="True Station")
# fig.plot(x=[best_lon], y=[best_lat], style="x0.4c", fill="red", pen="black", label="Inverted Station")
fig.plot(x=[LON_CEN], y=[LAT_CEN], style="t0.3c", fill="orange", pen="black", label="Initial Estimate")

# Label each event with idx and magnitude
for idx, (lon, lat, mag) in enumerate(zip(event_lons, event_lats, event_mags)):
    fig.text(x=lon, y=lat, text=f"{idx} (M{mag:.1f})", font="8p,Helvetica-Bold,black", justify="CM")


fig.legend(position="JBR+o0.2c", box=True)
fig.savefig("station_mercator_map.png")
print("Saved: station_mercator_map.png")


# fig = pygmt.Figure()
# fig.basemap(region=region, projection="M8i", frame=["af", f"+tStation Inversion: {STATION}"])
# fig.grdimage(grid="@earth_relief_01m", region=region, projection="M8i", shading=True)

# fig.plot(x=event_lons, y=event_lats, style="c0.15c", fill="blue", pen="black", label="Events")
# fig.plot(x=[true_lon], y=[true_lat], style="a0.3c", fill="green", pen="black", label="True Station")
# fig.plot(x=[best_lon], y=[best_lat], style="x0.4c", fill="red", pen="black", label="Inverted Station")
# fig.plot(x=[LON_CEN], y=[LAT_CEN], style="t0.3c", fill="orange", pen="black", label="Initial Estimate")

# for elat, elon in zip(event_lats, event_lons):
#     fig.plot(data=[[elon, elat], [best_lon, best_lat]], pen="0.3p,gray,-")

# station_code = f"{NETWORK}.{STATION}"
# fig.text(x=true_lon, y=true_lat, text=f"{station_code} (true)", font="8p,Helvetica-Bold,green", justify="TR", offset="0.2c/0.2c")
# fig.text(x=best_lon, y=best_lat, text="Inverted", font="8p,Helvetica-Bold,red", justify="BL", offset="0.2c/0.2c")
# fig.text(x=LON_CEN, y=LAT_CEN, text="Initial", font="8p,Helvetica-Bold,orange", justify="TL", offset="0.2c/0.2c")

# misfit_km = np.sqrt((best_lat - true_lat)**2 + (best_lon - true_lon)**2 + (best_z - true_elv)**2)
# fig.text(x=lon_min + 0.5, y=lat_min + 0.5,
#          text=f"Misfit: {misfit_km:.2f}°",
#          font="10p,Helvetica-Bold,black", justify="BL", offset="0.1c/0.1c")

# fig.legend(position="JBR+o0.2c", box=True)
# fig.savefig("station_mercator_map.png")
# print("Saved: station_mercator_map.png")













# # ========================================================================== 80
# # Picking
# # evt_idx = 8


# if len(st_win[0].data) < 10:
#     print(f"[Warning] Data too short for {evt_idx}")
#     # return None

# # if PICK_METHOD == "stalta":
# #     st_win.filter("bandpass", freqmin=FILTER[0], freqmax=FILTER[1], corners=4, zerophase=True)



# if PICK_METHOD == "hilbert":
#     abs_env = np.abs(hilbert(st_win[0].data))
#     peak_idx = np.argmax(abs_env)
#     if not peak_idx:
#         print(f"[Warning] No peak found for {evt_idx}")
#         # return None
#     t_obs = st_win.stats.starttime + peak_idx / st_win.stats.sampling_rate

# elif PICK_METHOD == "stalta":
#     nsta = int(STA * st_win.stats.sampling_rate)
#     nlta = int(LTA * st_win.stats.sampling_rate)
#     cft = classic_sta_lta(st_win.data, nsta, nlta)
#     on_off = trigger_onset(cft, TRIGGER_ON, TRIGGER_OFF)
#     if len(on_off) == 0:
#         print(f"[Warning] No trigger found for {evt_idx}")  
#         # return None
#     trigger_sample = on_off[0][0]
#     t_obs = st_win.stats.starttime + trigger_sample / st_win.stats.sampling_rate

# elif PICK_METHOD == "random":
#     t_obs = UTCDateTime(np.random.uniform(st_win[0].stats.starttime, st_win[0].stats.endtime))

# elif PICK_METHOD == "svd":
#     sr = st_win[0].stats.sampling_rate
#     z = st_win.select(component="Z")[0].data
#     n = st_win.select(component="1")[0].data
#     e = st_win.select(component="2")[0].data
#     times = st_win[0].times("utcdatetime")

#     win_len = int(0.2 * sr)
#     step = int(0.05 * sr)
#     threshold = 0.8

#     best_p = None
#     best_s = None

#     for i in range(0, len(z) - win_len, step):
#         z_win = z[i:i + win_len]
#         n_win = n[i:i + win_len]
#         e_win = e[i:i + win_len]
#         M = np.vstack([z_win, n_win, e_win])
#         U, S, Vt = np.linalg.svd(M, full_matrices=False)
#         linearity = S[0] / np.sum(S)
#         if linearity < threshold:
#             continue

#         vec = np.abs(U[:, 0])
#         dominant = np.argmax(vec)
#         if dominant == 0 and best_p is None:
#             best_p = times[i]
#         elif dominant in [1, 2] and best_s is None:
#             best_s = times[i]

#         if best_p and best_s:
#             break

#     if best_p is None:
#         print(f"[Warning] No P-wave found via SVD for {evt_idx}")
#         # return None

#     t_obs = best_p  # prioritize P-pick

# else:
#     print("Invalid PICK_METHOD")
#     # return None

# dist_deg = locations2degrees(event_lats[evt_idx], event_lons[evt_idx], inv[0][0].latitude, inv[0][0].longitude)
# arrivals = model.get_travel_times(source_depth_in_km=event_depths[evt_idx], distance_in_degree=dist_deg, phase_list=["P"])
# if not arrivals:
#     print(f"[Warning] No arrivals found for {evt_idx}")
#     # return None
# t_pred = origin_times[evt_idx] + arrivals[0].time

# # ========================================================================== 80
# # Run polarization analysis
# times_paz, rect, incidence = compute_polarization_manual(st_win)

# # Grab the Z trace
# trZ = st_win.select(component="Z")[0]
# trN = st_win.select(component="1")[0]
# trE = st_win.select(component="2")[0]


# # Run the plotting function
# p_pick, s_pick = plot_polarization_result(
#     trZ=trZ,
#     trN=trN,
#     trE=trE,
#     times_paz=times_paz,
#     rect=rect,
#     incidence=incidence,
#     rect_thresh=0.7,
#     p_incidence_max=30,
#     s_incidence_min=45,
#     amp_thresh=0.01,
#     p_pick_search_start=5.0,
#     filename="polarization_output.png",
#     show_plot=False,
#     true_p_arrival=P_arrival_times[evt_idx],
#     true_s_arrival=S_arrival_times[evt_idx]
# )

# print(f"P-pick: {p_pick:.2f} s | S-pick: {s_pick:.2f} s")









#     # Plot and save waveform with pick and prediction
#     try:
# if PICK_METHOD == "svd":
#     # === Plotting polarization classification ===
#     fig, ax = plt.subplots(figsize=(8, 3))
#     t_series = np.linspace(0, len(z) / sr, len(z))
#     ax.plot(t_series, z, label="Z component", color="k", linewidth=0.8)

#     for idx in p_indices:
#         t0 = idx / sr
#         t1 = (idx + win_len) / sr
#         ax.axvspan(t0, t1, color="red", alpha=0.3, label="P-polarized" if idx == p_indices[0] else None)

#     for idx in s_indices:
#         t0 = idx / sr
#         t1 = (idx + win_len) / sr
#         ax.axvspan(t0, t1, color="blue", alpha=0.3, label="S-polarized" if idx == s_indices[0] else None)

#     ax.set_title(f"SVD Polarization: Event {evt_idx}")
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Amplitude (Z)")
#     ax.legend(loc="upper right")
#     plt.tight_layout()
#     fig.savefig(f"picks/waveform_plots/svd_event_{evt_idx:03d}.png")
#     plt.close(fig)                
        
#         else:    
#             fig, ax = plt.subplots(figsize=(8, 3))
#             times = np.linspace(0, st_win.stats.npts / st_win.stats.sampling_rate, st_win.stats.npts)
#             ax.plot(times, st_win.data, label="Waveform")
#             ax.axvline((t_obs - st_win.stats.starttime), color="r", linestyle="--", label="Pick")
#             ax.axvline((t_pred - st_win.stats.starttime), color="g", linestyle=":", label="Predicted")
#             ax.set_title(f"Event {evt_idx}: M={magnitude.mag:.1f}, Pick Method = {PICK_METHOD}")
#             ax.set_xlabel("Time (s)")
#             ax.set_ylabel("Amplitude")
#             ax.legend()
#             plt.tight_layout()
#             fig.savefig(f"picks/waveform_plots/event_{evt_idx:03d}.png")
#             plt.close(fig)
        
#     except Exception as e:
#         print(f"[Warning] Plotting failed for event {evt_idx}: {e}")

#     return (ev_lat, ev_lon, ev_depth, origin_time, t_obs)
# except:
#     print(f"[Error] Processing failed for event {evt_idx}")
#     # return None










# def process_event(event, evt_idx):
#     try:
#         origin = event.preferred_origin() or event.origins[0]
#         magnitude = event.preferred_magnitude() or event.magnitudes[0]
#         ev_lat, ev_lon, ev_depth = origin.latitude, origin.longitude, origin.depth / 1000
#         origin_time = origin.time
        
#         geod = Geod(ellps="WGS84")
#         # Find closest and furthest points in search grid for arrival time bounds
#         az12, az21, distances = geod.inv(
#             np.full_like(flat_lons, ev_lon),  # longitudes from point
#             np.full_like(flat_lats, ev_lat),  # latitudes from point
#             flat_lons,                      # destination longitudes
#             flat_lats                       # destination latitudes
#         )
        
#         # Find closest and furthest
#         closest_idx = np.argmin(distances)
#         furthest_idx = np.argmax(distances)

#         dist_deg_close = locations2degrees(ev_lat, ev_lon, flat_lats[closest_idx], flat_lons[closest_idx])
#         dist_deg_far = locations2degrees(ev_lat, ev_lon, flat_lats[furthest_idx], flat_lons[furthest_idx])
#         arrivals_close = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg_close, phase_list=["P"])
#         arrivals_far = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg_far, phase_list=["P"])        
        
#         if not arrivals_close:
#             # print(f"[Warning] No arrivals close found for {evt_idx}")
#             # return None
#             t_pred_start = origin_time
#         elif arrivals_close:
#             t_pred_start = origin_time + arrivals_close[0].time
        
#         if not arrivals_far:
#             print(f"[Warning] No arrivals far found for {evt_idx}")
#             return None

#         t_pred_end  = origin_time + arrivals_far[0].time        
         
#         st_win = tr.copy().trim(starttime=t_pred_start - pick_before, endtime=t_pred_end + pick_after)
#         st_win.write(f"picks/waveforms/event_{evt_idx:03d}.mseed", format="MSEED")
        
#         if len(st_win.data) < 10:
#             print(f"[Warning] Data too short for {evt_idx}")
#             return None
        
#         # if PICK_METHOD == "stalta":
#         #     st_win.filter("bandpass", freqmin=FILTER[0], freqmax=FILTER[1], corners=4, zerophase=True)

#         st_win.filter("bandpass", freqmin=FILTER[0], freqmax=FILTER[1], corners=4, zerophase=True)


#         if PICK_METHOD == "hilbert":
#             abs_env = np.abs(hilbert(st_win.data))
#             peak_idx = np.argmax(abs_env)
#             if not peak_idx:
#                 print(f"[Warning] No peak found for {evt_idx}")
#                 return None
#             t_obs = st_win.stats.starttime + peak_idx / st_win.stats.sampling_rate

#         elif PICK_METHOD == "stalta":
#             nsta = int(STA * st_win.stats.sampling_rate)
#             nlta = int(LTA * st_win.stats.sampling_rate)
#             cft = classic_sta_lta(st_win.data, nsta, nlta)
#             on_off = trigger_onset(cft, TRIGGER_ON, TRIGGER_OFF)
#             if len(on_off) == 0:
#                 print(f"[Warning] No trigger found for {evt_idx}")  
#                 return None
#             trigger_sample = on_off[0][0]
#             t_obs = st_win.stats.starttime + trigger_sample / st_win.stats.sampling_rate

#         elif PICK_METHOD == "random":
#             t_obs = UTCDateTime(np.random.uniform(st_win.stats.starttime, st_win.stats.endtime))

#         elif PICK_METHOD == "svd":
#             sr = st_win[0].stats.sampling_rate
#             z = st_win.select(component="Z")[0].data
#             n = st_win.select(component="N")[0].data
#             e = st_win.select(component="E")[0].data
#             times = st_win[0].times("utcdatetime")

#             win_len = int(0.2 * sr)
#             step = int(0.05 * sr)
#             threshold = 0.8

#             best_p = None
#             best_s = None

#             for i in range(0, len(z) - win_len, step):
#                 z_win = z[i:i + win_len]
#                 n_win = n[i:i + win_len]
#                 e_win = e[i:i + win_len]
#                 M = np.vstack([z_win, n_win, e_win])
#                 U, S, Vt = np.linalg.svd(M, full_matrices=False)
#                 linearity = S[0] / np.sum(S)
#                 if linearity < threshold:
#                     continue

#                 vec = np.abs(U[:, 0])
#                 dominant = np.argmax(vec)
#                 if dominant == 0 and best_p is None:
#                     best_p = times[i]
#                 elif dominant in [1, 2] and best_s is None:
#                     best_s = times[i]

#                 if best_p and best_s:
#                     break

#             if best_p is None:
#                 print(f"[Warning] No P-wave found via SVD for {evt_idx}")
#                 return None

#             t_obs = best_p  # prioritize P-pick

#         else:
#             print("Invalid PICK_METHOD")
#             return None
       
#         dist_deg = locations2degrees(ev_lat, ev_lon, inv[0][0].latitude, inv[0][0].longitude)
#         arrivals = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg, phase_list=["P"])
#         if not arrivals:
#             print(f"[Warning] No arrivals found for {evt_idx}")
#             return None
#         t_pred = origin_time + arrivals[0].time

#         # Plot and save waveform with pick and prediction
#         try:
#             if PICK_METHOD == "svd":
#                 # === Plotting polarization classification ===
#                 fig, ax = plt.subplots(figsize=(8, 3))
#                 t_series = np.linspace(0, len(z) / sr, len(z))
#                 ax.plot(t_series, z, label="Z component", color="k", linewidth=0.8)

#                 for idx in p_indices:
#                     t0 = idx / sr
#                     t1 = (idx + win_len) / sr
#                     ax.axvspan(t0, t1, color="red", alpha=0.3, label="P-polarized" if idx == p_indices[0] else None)

#                 for idx in s_indices:
#                     t0 = idx / sr
#                     t1 = (idx + win_len) / sr
#                     ax.axvspan(t0, t1, color="blue", alpha=0.3, label="S-polarized" if idx == s_indices[0] else None)

#                 ax.set_title(f"SVD Polarization: Event {evt_idx}")
#                 ax.set_xlabel("Time (s)")
#                 ax.set_ylabel("Amplitude (Z)")
#                 ax.legend(loc="upper right")
#                 plt.tight_layout()
#                 fig.savefig(f"picks/waveform_plots/svd_event_{evt_idx:03d}.png")
#                 plt.close(fig)                
            
#             else:    
#                 fig, ax = plt.subplots(figsize=(8, 3))
#                 times = np.linspace(0, st_win.stats.npts / st_win.stats.sampling_rate, st_win.stats.npts)
#                 ax.plot(times, st_win.data, label="Waveform")
#                 ax.axvline((t_obs - st_win.stats.starttime), color="r", linestyle="--", label="Pick")
#                 ax.axvline((t_pred - st_win.stats.starttime), color="g", linestyle=":", label="Predicted")
#                 ax.set_title(f"Event {evt_idx}: M={magnitude.mag:.1f}, Pick Method = {PICK_METHOD}")
#                 ax.set_xlabel("Time (s)")
#                 ax.set_ylabel("Amplitude")
#                 ax.legend()
#                 plt.tight_layout()
#                 fig.savefig(f"picks/waveform_plots/event_{evt_idx:03d}.png")
#                 plt.close(fig)
            
#         except Exception as e:
#             print(f"[Warning] Plotting failed for event {evt_idx}: {e}")

#         return (ev_lat, ev_lon, ev_depth, origin_time, t_obs)
#     except:
#         print(f"[Error] Processing failed for event {evt_idx}")
#         return None

# with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
#     futures = [executor.submit(process_event, evt, idx) for idx, evt in enumerate(cat)]
#     for f in tqdm(as_completed(futures), total=len(cat), desc="Picking arrivals"):
#         result = f.result()
#         if result:
#             ev_lat, ev_lon, ev_depth, origin_time, t_obs = result
#             event_lats.append(ev_lat)
#             event_lons.append(ev_lon)
#             event_depths.append(ev_depth)
#             origin_times.append(origin_time)
#             observed_arrivals.append(t_obs)


# # === Misfit evaluation ===
# def compute_misfit_for_location(lat, lon, elev_km):
#     residuals = []
#     for ev_lat, ev_lon, ev_depth, t0_evt, t_obs in zip(event_lats, event_lons, event_depths, origin_times, observed_arrivals):
#         dist_deg = locations2degrees(ev_lat, ev_lon, lat, lon)
#         arrivals = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg, phase_list=["P"])
#         if not arrivals:
#             continue
#         t_model = arrivals[0].time - elev_km / arrivals[0].ray_param
#         t_pred = t0_evt + t_model
#         residuals.append(t_obs - t_pred)
#     return np.mean(np.square(residuals)) if residuals else np.inf

# # === Inversion ===
# if INVERSION_METHOD == "gradient":
#     print("Running gradient-free optimization with parallel misfit evaluations...")

#     history = []
#     call_counter = [0]
#     pbar = tqdm(total=num_iter * 10, desc="Nelder-Mead Evaluations", dynamic_ncols=True)

#     def objective_with_tracking(x):
#         if isinstance(x[0], (list, np.ndarray)):
#             with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
#                 futures = [executor.submit(compute_misfit_for_location, *xi) for xi in x]
#                 results = [f.result() for f in futures]
#                 history.extend(results)
#                 call_counter[0] += 1
#                 pbar.update(1)
#                 return np.array(results)
#         else:
#             misfit = compute_misfit_for_location(*x)
#             history.append(misfit)
#             return misfit

#     result = minimize(objective_with_tracking,
#                       x0=[LAT_CEN, LON_CEN, 1.0],
#                       method="Nelder-Mead",
#                       bounds=[LAT_RANGE, LON_RANGE, Z_RANGE],
#                       options={"maxiter": num_iter})
#     pbar.close()
    
#     best_lat, best_lon, best_z = result.x

#     # === Plot convergence ===
#     plt.figure(figsize=(8, 5))
#     plt.plot(history, marker="o", lw=1)
#     plt.title("Nelder-Mead Convergence")
#     plt.xlabel("Iteration")
#     plt.ylabel("Misfit (L2)")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("gradient_convergence.png")
#     print("Saved: gradient_convergence.png")


# elif INVERSION_METHOD == "grid+mcmc":
#     print("Running coarse grid search...")
#     lat_coarse = np.linspace(*LAT_RANGE, N_LAT_COURSE)
#     lon_coarse = np.linspace(*LON_RANGE, N_LON_COURSE)
#     z_coarse   = np.linspace(*Z_RANGE, N_Z_COURSE)

#     best_misfit = np.inf
#     tasks = [(lat, lon, z) for lat in lat_coarse for lon in lon_coarse for z in z_coarse]
#     with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
#         futures = {executor.submit(compute_misfit_for_location, lat, lon, z): (lat, lon, z) for lat, lon, z in tasks}
#         for f in tqdm(as_completed(futures), total=len(tasks), desc="Grid Search"):
#             misfit = f.result()
#             lat, lon, z = futures[f]
#             if misfit < best_misfit:
#                 best_misfit = misfit
#                 best_lat, best_lon, best_z = lat, lon, z

#     print(f"Coarse estimate: lat={best_lat:.3f}, lon={best_lon:.3f}, z={best_z:.2f} km")

#     print("Running MCMC refinement...")
#     def log_prob(x):
#         lat, lon, z = x
#         if not (LAT_RANGE[0] <= lat <= LAT_RANGE[1] and
#                 LON_RANGE[0] <= lon <= LON_RANGE[1] and
#                 Z_RANGE[0] <= z <= Z_RANGE[1]):
#             return -np.inf
#         return -compute_misfit_for_location(lat, lon, z)

#     ndim = 3
#     nwalkers = 16
#     p0 = [np.array([best_lat, best_lon, best_z]) + 0.05 * np.random.randn(ndim) for _ in range(nwalkers)]
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
#     sampler.run_mcmc(p0, 200, progress=True)

#     flat_samples = sampler.get_chain(discard=50, thin=10, flat=True)
#     best_idx = np.argmin([compute_misfit_for_location(*loc) for loc in flat_samples])
#     best_lat, best_lon, best_z = flat_samples[best_idx]

#     # === MCMC Convergence Plot ===
#     samples = sampler.get_chain()
#     fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
#     labels = ["Latitude", "Longitude", "Elevation (km)"]
#     for i in range(3):
#         axes[i].plot(samples[:, :, i], alpha=0.4)
#         axes[i].set_ylabel(labels[i])
#     axes[-1].set_xlabel("Step Number")
#     plt.suptitle("MCMC Walker Convergence")
#     plt.tight_layout()
#     plt.savefig("mcmc_convergence.png")

#     try:
#         import corner
#         fig = corner.corner(flat_samples, labels=labels, truths=[best_lat, best_lon, best_z])
#         fig.savefig("mcmc_corner.png")
#     except ImportError:
#         print("Install `corner` to get corner plot.")

# # === Output ===
# true_lat = inv[0][0].latitude
# true_lon = inv[0][0].longitude
# true_elv = inv[0][0].elevation / 1000.0

# print(f"Estimated station: lat={best_lat:.3f}, lon={best_lon:.3f}, elev={best_z:.2f} km")
# print(f"True station:      lat={true_lat:.3f}, lon={true_lon:.3f}, elev={true_elv:.2f} km")

# # === Misfit surface plot (lat-lon at best_z) ===
# # print("Generating misfit surface plot...")

# # lat_vals = np.linspace(LAT_RANGE[0], LAT_RANGE[1], 100)
# # lon_vals = np.linspace(LON_RANGE[0], LON_RANGE[1], 100)
# # misfit_surface = np.zeros((len(lat_vals), len(lon_vals)))

# # with tqdm(total=len(lat_vals) * len(lon_vals), desc="Evaluating Surface") as pbar:
# #     for i, lat in enumerate(lat_vals):
# #         for j, lon in enumerate(lon_vals):
# #             misfit_surface[i, j] = compute_misfit_for_location(lat, lon, best_z)
# #             pbar.update(1)

# # LAT, LON = np.meshgrid(lon_vals, lat_vals)

# # plt.figure(figsize=(10, 6))
# # cs = plt.contourf(LON, LAT, misfit_surface, levels=30, cmap='viridis')
# # plt.colorbar(cs, label='Misfit (s²)')
# # plt.plot(true_lon, true_lat, 'go', label='True Location')
# # plt.plot(best_lon, best_lat, 'r*', markersize=12, label='Inverted Location')
# # plt.plot(LON_CEN, LAT_CEN, 'c^', markersize=8, label='Initial Guess')
# # plt.xlabel("Longitude")
# # plt.ylabel("Latitude")
# # plt.title(f"Misfit Surface at Depth {best_z:.2f} km")
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig("misfit_surface_map.png")
# # plt.close()
# # print("Saved: misfit_surface_map.png")

# # === GMT Mercator Map ===
# print("Plotting GMT Mercator map...")
# lat_min_data = min(event_lats + [true_lat, best_lat]) - 2
# lat_max_data = max(event_lats + [true_lat, best_lat]) + 2
# lon_min_data = min(event_lons + [true_lon, best_lon]) - 2
# lon_max_data = max(event_lons + [true_lon, best_lon]) + 2

# lat_min_input, lat_max_input = LAT_RANGE
# lon_min_input, lon_max_input = LON_RANGE

# lat_min = min(lat_min_data, lat_min_input)
# lat_max = max(lat_max_data, lat_max_input)
# lon_min = min(lon_min_data, lon_min_input)
# lon_max = max(lon_max_data, lon_max_input)

# region = [lon_min, lon_max, lat_min, lat_max]


# fig = pygmt.Figure()
# fig.basemap(region=region, projection="M6i", frame=["af", f"+tStation Inversion: {STATION}"])
# fig.grdimage(grid="@earth_relief_01m", region=region, projection="M6i", shading=True)

# # Plot translucent grid box showing inversion search area
# fig.plot(
#     x=[LON_RANGE[0], LON_RANGE[1], LON_RANGE[1], LON_RANGE[0], LON_RANGE[0]],
#     y=[LAT_RANGE[0], LAT_RANGE[0], LAT_RANGE[1], LAT_RANGE[1], LAT_RANGE[0]],
#     pen="1p,gray",
#     transparency=70,
#     fill="white"
# )

# # Draw grid lines inside search area
# lon_ticks = np.arange(LON_RANGE[0], LON_RANGE[1] + 1e-5, 1.0)
# lat_ticks = np.arange(LAT_RANGE[0], LAT_RANGE[1] + 1e-5, 1.0)
# for lon in lon_ticks:
#     fig.plot(x=[lon, lon], y=[LAT_RANGE[0], LAT_RANGE[1]], pen="0.25p,gray,-")
# for lat in lat_ticks:
#     fig.plot(x=[LON_RANGE[0], LON_RANGE[1]], y=[lat, lat], pen="0.25p,gray,-")

# # Plot events and stations
# fig.plot(x=event_lons, y=event_lats, style="c0.15c", fill="blue", pen="black", label="Events")
# fig.plot(x=[true_lon], y=[true_lat], style="a0.3c", fill="green", pen="black", label="True Station")
# fig.plot(x=[best_lon], y=[best_lat], style="x0.4c", fill="red", pen="black", label="Inverted Station")
# fig.plot(x=[LON_CEN], y=[LAT_CEN], style="t0.3c", fill="orange", pen="black", label="Initial Estimate")

# for elat, elon in zip(event_lats, event_lons):
#     fig.plot(data=[[elon, elat], [best_lon, best_lat]], pen="0.3p,gray,-")

# # Misfit value annotation
# misfit_km = np.sqrt((best_lat - true_lat)**2 + (best_lon - true_lon)**2 + (best_z - true_elv)**2)
# fig.text(
#     x=lon_min + 0.5, y=lat_min + 0.5,
#     text=f"Misfit: {misfit_km:.2f}°",
#     font="10p,Helvetica-Bold,black", justify="BL", offset="0.1c/0.1c")

# fig.legend(position="JBR+o0.2c", box=True)
# fig.savefig("station_mercator_map.png")
# print("Saved: station_mercator_map.png")


# # fig = pygmt.Figure()
# # fig.basemap(region=region, projection="M8i", frame=["af", f"+tStation Inversion: {STATION}"])
# # fig.grdimage(grid="@earth_relief_01m", region=region, projection="M8i", shading=True)

# # fig.plot(x=event_lons, y=event_lats, style="c0.15c", fill="blue", pen="black", label="Events")
# # fig.plot(x=[true_lon], y=[true_lat], style="a0.3c", fill="green", pen="black", label="True Station")
# # fig.plot(x=[best_lon], y=[best_lat], style="x0.4c", fill="red", pen="black", label="Inverted Station")
# # fig.plot(x=[LON_CEN], y=[LAT_CEN], style="t0.3c", fill="orange", pen="black", label="Initial Estimate")

# # for elat, elon in zip(event_lats, event_lons):
# #     fig.plot(data=[[elon, elat], [best_lon, best_lat]], pen="0.3p,gray,-")

# # station_code = f"{NETWORK}.{STATION}"
# # fig.text(x=true_lon, y=true_lat, text=f"{station_code} (true)", font="8p,Helvetica-Bold,green", justify="TR", offset="0.2c/0.2c")
# # fig.text(x=best_lon, y=best_lat, text="Inverted", font="8p,Helvetica-Bold,red", justify="BL", offset="0.2c/0.2c")
# # fig.text(x=LON_CEN, y=LAT_CEN, text="Initial", font="8p,Helvetica-Bold,orange", justify="TL", offset="0.2c/0.2c")

# # misfit_km = np.sqrt((best_lat - true_lat)**2 + (best_lon - true_lon)**2 + (best_z - true_elv)**2)
# # fig.text(x=lon_min + 0.5, y=lat_min + 0.5,
# #          text=f"Misfit: {misfit_km:.2f}°",
# #          font="10p,Helvetica-Bold,black", justify="BL", offset="0.1c/0.1c")

# # fig.legend(position="JBR+o0.2c", box=True)
# # fig.savefig("station_mercator_map.png")
# # print("Saved: station_mercator_map.png")

# # === Plot arrival picks ===
# print("Plotting arrival picks...")
# fig, ax = plt.subplots(figsize=(12, 6))
# for ev_lat, ev_lon, t_obs, t0_evt in zip(event_lats, event_lons, observed_arrivals, origin_times):
#     delta = (t_obs - t0_evt)
#     ax.plot(delta, ev_lat, 'bo')

# ax.set_xlabel("Predicted Arrival Time after Origin (s)")
# ax.set_ylabel("Event Latitude")
# ax.set_title("Picked P-wave Arrival Delays by Event")
# plt.grid(True)
# plt.tight_layout()
# plt.savefig("arrival_picks.png")
# print("Saved: arrival_picks.png")


# waveforms = read("picks/waveforms/*.mseed")
# waveforms_noir = waveforms.copy().remove_response(inventory=inv, output="VEL")

# waveforms_filtered = waveforms.copy().filter(type="bandpass", freqmin=FILTER[0], freqmax=FILTER[1])
# waveforms_noir_filtered = waveforms_noir.copy().filter(type="bandpass", freqmin=FILTER[0], freqmax=FILTER[1])