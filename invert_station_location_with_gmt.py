import numpy as np
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from obspy import UTCDateTime, read, read_inventory
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from scipy.signal import hilbert
from concurrent.futures import ThreadPoolExecutor, as_completed
import pygmt
import warnings
warnings.simplefilter("ignore")

# === Configuration ===
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

# === Arrival Picker Configuration ===
PICK_METHOD = "stalta"  # "hilbert" or "stalta"
FILTER = (0.5, 5.0)
STA = 1.0
LTA = 10.0
TRIGGER_ON = 3.5
TRIGGER_OFF = 0.5

# === Time Window ===
t0 = UTCDateTime(DATE)
t1 = t0 + 86400

# === Validate waveform ===
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

# === Load waveform and response ===
client = Client("IRIS")
if waveform_is_valid(WAVEFORM_FILE, NETWORK, STATION, LOCATION, CHANNEL, t0, t1):
    print("Loading waveform from cache...")
    st = read(WAVEFORM_FILE)
else:
    print("Downloading waveform...")
    st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNEL, t0, t1)
    st.write(WAVEFORM_FILE, format="MSEED")

if os.path.exists(RESPONSE_FILE):
    inv = read_inventory(RESPONSE_FILE)
else:
    inv = client.get_stations(network=NETWORK, station=STATION, location=LOCATION,
                              channel=CHANNEL, level="response", starttime=t0)
    inv.write(RESPONSE_FILE, format="STATIONXML")

tr = st[0]
model = TauPyModel(MODEL)

# === Fetch catalog ===
cat = client.get_events(starttime=t0, endtime=t1, minmagnitude=MIN_MAG,
                        latitude=LAT_CEN, longitude=LON_CEN, maxradius=MAXRADIUS)

# === Arrival picking ===
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
        raise ValueError("Invalid PICK_METHOD")

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
    plt.savefig(f"picks/pick_{origin_time.strftime('%Y%m%d_%H%M%S')}.png")
    plt.close()

# === Inversion grid ===
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
        arrivals = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg, phase_list=["P"])
        if not arrivals:
            continue
        t_model = arrivals[0].time - elev_km / arrivals[0].ray_param
        t_pred = t0_evt + t_model
        residuals.append(t_obs - t_pred)
    return (i, j, k, np.mean(np.square(residuals)) if residuals else np.nan)

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
true_lat = inv[0][0].latitude
true_lon = inv[0][0].longitude
true_elv = inv[0][0].elevation / 1000.0

print(f"\nEstimated station: lat={best_lat:.3f}, lon={best_lon:.3f}, elev={best_z:.2f} km")
print(f"True station:      lat={true_lat:.3f}, lon={true_lon:.3f}, elev={true_elv:.2f} km")

# === Misfit surface plot ===
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

# === GMT Globe Map (zoomed with relief, arcs, labels) ===
print("Plotting GMT globe map...")

mid_lat = np.mean(event_lats + [best_lat, true_lat])
mid_lon = np.mean(event_lons + [best_lon, true_lon])
region = [mid_lon - 5, mid_lon + 5, mid_lat - 5, mid_lat + 5]

fig = pygmt.Figure()
fig.basemap(region="g", projection=f"E{mid_lon}/{mid_lat}/6i", frame=True)
fig.grdimage(grid="@earth_relief_01m", shading=True)


# Plot events
fig.plot(x=event_lons, y=event_lats, style="c0.15c", fill="blue", pen="black", label="Events")

# Plot station locations
fig.plot(x=[true_lon], y=[true_lat], style="a0.3c", fill="green", pen="black", label="True Station")
fig.plot(x=[best_lon], y=[best_lat], style="x0.4c", fill="red", pen="black", label="Inverted Station")

# Great-circle arcs from events to inverted location
for elat, elon in zip(event_lats, event_lons):
    fig.plot(data=[[elon, elat], [best_lon, best_lat]], pen="0.3p,gray,-")

# Labels
station_code = f"{NETWORK}.{STATION}"
fig.text(x=true_lon, y=true_lat, text=f"{station_code} (true)", font="10p,Helvetica-Bold,green", justify="TR", offset="0.2c/0.2c")
fig.text(x=best_lon, y=best_lat, text="Inverted", font="10p,Helvetica-Bold,red", justify="BL", offset="0.2c/0.2c")

# Label up to 5 events by origin time
for ev_lat, ev_lon, t0_evt in zip(event_lats[:5], event_lons[:5], origin_times[:5]):
    label = t0_evt.strftime("%H:%M")
    fig.text(x=ev_lon, y=ev_lat, text=label, font="7p,Helvetica,black", justify="TL", offset="0.1c/0.1c")

fig.legend(position="JBR+o0.2c", box=True)
fig.savefig("station_globe_map.png")
print("Saved map to station_globe_map.png")
