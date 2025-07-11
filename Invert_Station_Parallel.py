import numpy as np
import matplotlib.pyplot as plt
import os
from obspy import UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import hilbert
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.simplefilter("ignore")
import matplotlib
matplotlib.use("Agg")  # safer for parallel execution

# === CONFIGURATION ===
NETWORK = "IU"
STATION = "ANMO"
CHANNEL = "BHZ"
LOCATION = "00"
DATE = "2019-07-06"
MIN_MAG = 5.5
LAT_CEN, LON_CEN = 34.9, -106.5
MAXRADIUS = 90
GRID_STEP = 0.2
Z_STEP = 1.0
LAT_RANGE = (33.0, 36.5)
LON_RANGE = (-108.0, -104.0)
Z_RANGE = (0.0, 3.0)
MODEL = "iasp91"
N_WORKERS = 8
WAVEFORM_FILE = "cached_waveform.mseed"
RESPONSE_FILE = "cached_response.xml"
os.makedirs("picks", exist_ok=True)

# === Arrival Picking Options ===
PICK_METHOD = "hilbert"     # Options: "hilbert" or "stalta"
FILTER = (0.5, 5.0)         # Bandpass filter in Hz (used only with STA/LTA)
STA = 1.0                   # STA window in seconds
LTA = 10.0                  # LTA window in seconds
TRIGGER_ON = 3.5
TRIGGER_OFF = 0.5

# === TIME WINDOW ===
t0 = UTCDateTime(DATE)
t1 = t0 + 86400

# === VALIDATOR FOR WAVEFORM ===
def waveform_is_valid(filename, network, station, location, channel, starttime, endtime, min_bytes=1_000_000):
    try:
        if not os.path.exists(filename) or os.path.getsize(filename) < min_bytes:
            return False
        st = read(filename)
        if len(st) != 1:
            return False
        tr = st[0]
        if not (tr.stats.network == network and
                tr.stats.station == station and
                tr.stats.channel == channel and
                tr.stats.location.strip() == location.strip()):
            return False
        start_ok = abs(tr.stats.starttime - starttime) < 10
        end_ok = abs(tr.stats.endtime - endtime) < 10
        return start_ok and end_ok
    except Exception:
        return False

# === FETCH WAVEFORM ===
client = Client("IRIS")
if waveform_is_valid(WAVEFORM_FILE, NETWORK, STATION, LOCATION, CHANNEL, t0, t1):
    print("Loading waveform from cache...")
    st = read(WAVEFORM_FILE)
else:
    print("Downloading waveform...")
    st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNEL, t0, t1)
    st.write(WAVEFORM_FILE, format="MSEED")
    print(f"Saved waveform to: {WAVEFORM_FILE}")

# === FETCH RESPONSE ===
if os.path.exists(RESPONSE_FILE):
    print("Loading station response from cache...")
    inv = read_inventory(RESPONSE_FILE)
else:
    print("Downloading station response...")
    inv = client.get_stations(network=NETWORK, station=STATION, location=LOCATION,
                              channel=CHANNEL, level="response", starttime=t0)
    inv.write(RESPONSE_FILE, format="STATIONXML")

# === PREPARE DATA ===
print("Using raw data with instrument response intact.")
tr = st[0]
model = TauPyModel(MODEL)

# === FETCH CATALOG ===
print("Fetching events...")
cat = client.get_events(starttime=t0, endtime=t1, minmagnitude=MIN_MAG,
                        latitude=LAT_CEN, longitude=LON_CEN, maxradius=MAXRADIUS)

# === PICK P ARRIVALS ===
event_lats, event_lons, event_depths, origin_times, observed_arrivals = [], [], [], [], []
for event in cat:
    origin = event.preferred_origin() or event.origins[0]
    magnitude = event.preferred_magnitude() or event.magnitudes[0]
    ev_lat, ev_lon, ev_depth = origin.latitude, origin.longitude, origin.depth / 1000
    origin_time = origin.time

    dist_deg = locations2degrees(ev_lat, ev_lon, inv[0][0].latitude, inv[0][0].longitude)
    arrivals = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg, phase_list=["P"])
    if not arrivals:
        continue
    arrival = arrivals[0]
    t_pred = origin_time + arrival.time
    st_win = tr.copy().trim(starttime=t_pred - 5, endtime=t_pred + 15)
    if len(st_win.data) < 10:
        continue

    if PICK_METHOD == "stalta":
        st_win.filter("bandpass", freqmin=FILTER[0], freqmax=FILTER[1], corners=4, zerophase=True)

    if PICK_METHOD == "hilbert":
        abs_env = np.abs(hilbert(st_win.data))
        peak_idx = np.argmax(abs_env)
        t_obs = st_win.stats.starttime + peak_idx / st_win.stats.sampling_rate

    elif PICK_METHOD == "stalta":
        nsta = int(STA * st_win.stats.sampling_rate)
        nlta = int(LTA * st_win.stats.sampling_rate)
        cft = classic_sta_lta(st_win.data, nsta, nlta)
        on_off = trigger_onset(cft, TRIGGER_ON, TRIGGER_OFF)
        if len(on_off) == 0:
            continue
        trigger_sample = on_off[0][0]
        t_obs = st_win.stats.starttime + trigger_sample / st_win.stats.sampling_rate
    else:
        raise ValueError(f"Unknown pick method: {PICK_METHOD}")

    event_lats.append(ev_lat)
    event_lons.append(ev_lon)
    event_depths.append(ev_depth)
    origin_times.append(origin_time)
    observed_arrivals.append(t_obs)

    times = np.linspace(-5, 15, len(st_win.data))
    plt.figure(figsize=(10, 4))
    plt.plot(times, st_win.data, label="Velocity", color="black", lw=0.8)
    if PICK_METHOD == "hilbert":
        plt.plot(times, abs_env, label="Envelope", color="gray", lw=0.8)
    plt.axvline((t_pred - st_win.stats.starttime), color='red', ls='--', label='Predicted P')
    plt.axvline((t_obs - st_win.stats.starttime), color='blue', ls='--', label='Picked P')
    plt.title(f"{PICK_METHOD.upper()} pick: {origin_time.date} {origin_time.time:.2f} | M={magnitude.mag:.1f}")
    plt.xlabel("Time (s since window start)")
    plt.ylabel("Amplitude")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fname = f"picks/pick_{origin_time.strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(fname)
    plt.close()

print(f"Used {len(observed_arrivals)} arrivals.")

# === INVERSION GRID ===
lat_grid = np.arange(LAT_RANGE[0], LAT_RANGE[1], GRID_STEP)
lon_grid = np.arange(LON_RANGE[0], LON_RANGE[1], GRID_STEP)
z_grid   = np.arange(Z_RANGE[0], Z_RANGE[1], Z_STEP)
misfit_grid = np.full((len(lat_grid), len(lon_grid), len(z_grid)), np.nan)

def compute_misfit(i, j, k):
    lat = lat_grid[i]
    lon = lon_grid[j]
    elev_km = z_grid[k]
    residuals = []
    for ev_lat, ev_lon, ev_depth, t0_evt, t_obs in zip(event_lats, event_lons, event_depths, origin_times, observed_arrivals):
        dist_deg = locations2degrees(ev_lat, ev_lon, lat, lon)
        arrivals = model.get_travel_times(source_depth_in_km=ev_depth,
                                          distance_in_degree=dist_deg, phase_list=["P"])
        if not arrivals:
            continue
        t_model = arrivals[0].time - elev_km / arrivals[0].ray_param
        t_pred = t0_evt + t_model
        residuals.append(t_obs - t_pred)
    if residuals:
        return (i, j, k, np.mean(np.square(residuals)))
    else:
        return (i, j, k, np.nan)

print("Running parallel grid search...")
tasks = [(i, j, k) for i in range(len(lat_grid))
                  for j in range(len(lon_grid))
                  for k in range(len(z_grid))]

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = [executor.submit(compute_misfit, i, j, k) for i, j, k in tasks]
    for future in as_completed(futures):
        i, j, k, misfit = future.result()
        misfit_grid[i, j, k] = misfit

idx_min = np.unravel_index(np.nanargmin(misfit_grid), misfit_grid.shape)
best_lat, best_lon, best_z = lat_grid[idx_min[0]], lon_grid[idx_min[1]], z_grid[idx_min[2]]

print(f"\nEstimated station location:")
print(f"  Latitude:  {best_lat:.3f}")
print(f"  Longitude: {best_lon:.3f}")
print(f"  Elevation: {best_z:.2f} km")

true_lat = inv[0][0].latitude
true_lon = inv[0][0].longitude
true_elv = inv[0][0].elevation / 1000.0
print(f"True location:")
print(f"  Latitude:  {true_lat:.3f}")
print(f"  Longitude: {true_lon:.3f}")
print(f"  Elevation: {true_elv:.2f} km")
print(f"Angular error: {locations2degrees(best_lat, best_lon, true_lat, true_lon):.2f}°")

plt.figure(figsize=(10, 6))
plt.contourf(lon_grid, lat_grid, np.nanmin(misfit_grid, axis=2), levels=30)
plt.colorbar(label="L2 Misfit (s²)")
plt.plot(true_lon, true_lat, "go", label="True")
plt.plot(best_lon, best_lat, "r*", label="Estimate", markersize=12)
plt.title("Grid Search Misfit Surface (Lat-Lon slice)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.tight_layout()
plt.savefig("misfit_surface.png")
print("Saved misfit surface to misfit_surface.png")
