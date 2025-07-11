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
from scipy.optimize import minimize
import emcee
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pygmt
import warnings
warnings.simplefilter("ignore")

# === Configuration ===
NETWORK = "IU"
STATION = "ANMO"
CHANNEL = "BHZ"
LOCATION = "00"
DATE = "2025-07-01"
MIN_MAG = 2.0
LAT_CEN, LON_CEN = 30.0, -100.0
MAXRADIUS = 50
LAT_RANGE = (30.0, 40.0)
LON_RANGE = (-110.0, -100.0)
Z_RANGE = (0.0, 3.0)
MODEL = "iasp91"
# INVERSION_METHOD = "grid+mcmc"  # or "gradient"
INVERSION_METHOD = "gradient"
num_iter = 20
N_WORKERS = 8
WAVEFORM_FILE = "cached_waveform.mseed"
RESPONSE_FILE = "cached_response.xml"
os.makedirs("picks", exist_ok=True)

# Coarse grid resolution
N_LAT_COURSE = 6
N_LON_COURSE = 6
N_Z_COURSE = 4

# === Arrival Picker Configuration ===
PICK_METHOD = "hilbert"
# PICK_METHOD = "stalta"
FILTER = (0.5, 5.0)
STA = 1.0
LTA = 10.0
TRIGGER_ON = 3.5
TRIGGER_OFF = 0.5
pick_before = -5
pick_after = 15

# === Load waveform and response ===
t0 = UTCDateTime(DATE)
t1 = t0 + 86400

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

# === Phase picking ===
tr = st[0]
model = TauPyModel(MODEL)
cat = client.get_events(starttime=t0, endtime=t1, minmagnitude=MIN_MAG,
                        latitude=LAT_CEN, longitude=LON_CEN, maxradius=MAXRADIUS)

event_lats, event_lons, event_depths, origin_times, observed_arrivals = [], [], [], [], []

def process_event(event):
    try:
        origin = event.preferred_origin() or event.origins[0]
        magnitude = event.preferred_magnitude() or event.magnitudes[0]
        ev_lat, ev_lon, ev_depth = origin.latitude, origin.longitude, origin.depth / 1000
        origin_time = origin.time

        dist_deg = locations2degrees(ev_lat, ev_lon, inv[0][0].latitude, inv[0][0].longitude)
        arrivals = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg, phase_list=["P"])
        if not arrivals:
            return None
        arrival = arrivals[0]
        t_pred = origin_time + arrival.time
        st_win = tr.copy().trim(starttime=t_pred - pick_before, endtime=t_pred + pick_after)
        if len(st_win.data) < 10:
            return None
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
                return None
            trigger_sample = on_off[0][0]
            t_obs = st_win.stats.starttime + trigger_sample / st_win.stats.sampling_rate
        else:
            return None
        # times = np.linspace(-5, 15, len(st_win.data))
        # plt.figure(figsize=(10, 4))
        # plt.plot(times, st_win.data, label="Velocity", color="black", lw=0.8)
        # if PICK_METHOD == "hilbert":
        #     plt.plot(times, abs_env, label="Envelope", color="gray", lw=0.8)
        # plt.axvline((t_pred - st_win.stats.starttime), color='red', ls='--', label='Predicted P')
        # plt.axvline((t_obs - st_win.stats.starttime), color='blue', ls='--', label='Picked P')
        # plt.title(f"{PICK_METHOD.upper()} pick: {origin_time.date} {origin_time.time:.2f} | M={magnitude.mag:.1f}")
        # plt.xlabel("Time (s since window start)")
        # plt.ylabel("Amplitude")
        # plt.legend(loc="upper right", fontsize=8)
        # plt.tight_layout()
        # plt.savefig(f"picks/pick_{origin_time.strftime('%Y%m%d_%H%M%S')}.png")
        # plt.close()

        return (ev_lat, ev_lon, ev_depth, origin_time, t_obs)
    except:
        return None

with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
    futures = [executor.submit(process_event, evt) for evt in cat]
    for f in tqdm(as_completed(futures), total=len(cat), desc="Picking arrivals"):
        result = f.result()
        if result:
            ev_lat, ev_lon, ev_depth, origin_time, t_obs = result
            event_lats.append(ev_lat)
            event_lons.append(ev_lon)
            event_depths.append(ev_depth)
            origin_times.append(origin_time)
            observed_arrivals.append(t_obs)

# === Save arrival picks ===
print("Saving individual picks...")
with open("picks/arrival_times.txt", "w") as f:
    f.write("# Event_Lat  Event_Lon  Depth_km  OriginTime  ArrivalTime  Method\n")
    for ev_lat, ev_lon, ev_depth, t0_evt, t_obs in zip(event_lats, event_lons, event_depths, origin_times, observed_arrivals):
        f.write(f"{ev_lat:.3f}  {ev_lon:.3f}  {ev_depth:.2f}  {t0_evt.isoformat()}  {t_obs.isoformat()}  {PICK_METHOD}\n")

# === Save waveform plots for each pick ===
print("Saving waveform plots...")
os.makedirs("picks/waveform_plots", exist_ok=True)
for idx, (ev_lat, ev_lon, ev_depth, t0_evt, t_obs) in enumerate(zip(event_lats, event_lons, event_depths, origin_times, observed_arrivals)):
    dist_deg = locations2degrees(ev_lat, ev_lon, inv[0][0].latitude, inv[0][0].longitude)
    arrivals = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg, phase_list=["P"])
    if not arrivals:
        continue
    t_pred = t0_evt + arrivals[0].time
    st_win = tr.copy().trim(starttime=t_pred - 5, endtime=t_pred + 15)
    if len(st_win.data) < 10:
        continue

    fig, ax = plt.subplots(figsize=(8, 3))
    times = np.linspace(0, st_win.stats.npts / st_win.stats.sampling_rate, st_win.stats.npts)
    ax.plot(times, st_win.data, label="Waveform")
    ax.axvline((t_obs - st_win.stats.starttime), color="r", linestyle="--", label="Pick")
    ax.axvline((t_pred - st_win.stats.starttime), color="g", linestyle=":", label="Predicted")
    ax.set_title(f"Event {idx}: Pick Method = {PICK_METHOD}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.tight_layout()
    fig.savefig(f"picks/waveform_plots/event_{idx:03d}.png")
    plt.close(fig)

# === Misfit evaluation ===
def compute_misfit_for_location(lat, lon, elev_km):
    residuals = []
    for ev_lat, ev_lon, ev_depth, t0_evt, t_obs in zip(event_lats, event_lons, event_depths, origin_times, observed_arrivals):
        dist_deg = locations2degrees(ev_lat, ev_lon, lat, lon)
        arrivals = model.get_travel_times(source_depth_in_km=ev_depth, distance_in_degree=dist_deg, phase_list=["P"])
        if not arrivals:
            continue
        t_model = arrivals[0].time - elev_km / arrivals[0].ray_param
        t_pred = t0_evt + t_model
        residuals.append(t_obs - t_pred)
    return np.mean(np.square(residuals)) if residuals else np.inf

# === Inversion ===
if INVERSION_METHOD == "gradient":
    print("Running gradient-free optimization with parallel misfit evaluations...")

    history = []

    def objective_with_tracking(x):
        if isinstance(x[0], (list, np.ndarray)):
            with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
                futures = [executor.submit(compute_misfit_for_location, *xi) for xi in x]
                results = [f.result() for f in futures]
                history.extend(results)
                return np.array(results)
        else:
            misfit = compute_misfit_for_location(*x)
            history.append(misfit)
            return misfit

    result = minimize(objective_with_tracking,
                      x0=[LAT_CEN, LON_CEN, 1.0],
                      method="Nelder-Mead",
                      bounds=[LAT_RANGE, LON_RANGE, Z_RANGE],
                      options={"maxiter": num_iter})
    best_lat, best_lon, best_z = result.x

    # === Plot convergence ===
    plt.figure(figsize=(8, 5))
    plt.plot(history, marker="o", lw=1)
    plt.title("Nelder-Mead Convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Misfit (L2)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("gradient_convergence.png")
    print("Saved: gradient_convergence.png")


elif INVERSION_METHOD == "grid+mcmc":
    print("Running coarse grid search...")
    lat_coarse = np.linspace(*LAT_RANGE, N_LAT_COURSE)
    lon_coarse = np.linspace(*LON_RANGE, N_LON_COURSE)
    z_coarse   = np.linspace(*Z_RANGE, N_Z_COURSE)

    best_misfit = np.inf
    tasks = [(lat, lon, z) for lat in lat_coarse for lon in lon_coarse for z in z_coarse]
    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(compute_misfit_for_location, lat, lon, z): (lat, lon, z) for lat, lon, z in tasks}
        for f in tqdm(as_completed(futures), total=len(tasks), desc="Grid Search"):
            misfit = f.result()
            lat, lon, z = futures[f]
            if misfit < best_misfit:
                best_misfit = misfit
                best_lat, best_lon, best_z = lat, lon, z

    print(f"Coarse estimate: lat={best_lat:.3f}, lon={best_lon:.3f}, z={best_z:.2f} km")

    print("Running MCMC refinement...")
    def log_prob(x):
        lat, lon, z = x
        if not (LAT_RANGE[0] <= lat <= LAT_RANGE[1] and
                LON_RANGE[0] <= lon <= LON_RANGE[1] and
                Z_RANGE[0] <= z <= Z_RANGE[1]):
            return -np.inf
        return -compute_misfit_for_location(lat, lon, z)

    ndim = 3
    nwalkers = 16
    p0 = [np.array([best_lat, best_lon, best_z]) + 0.05 * np.random.randn(ndim) for _ in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(p0, 200, progress=True)

    flat_samples = sampler.get_chain(discard=50, thin=10, flat=True)
    best_idx = np.argmin([compute_misfit_for_location(*loc) for loc in flat_samples])
    best_lat, best_lon, best_z = flat_samples[best_idx]

    # === MCMC Convergence Plot ===
    samples = sampler.get_chain()
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    labels = ["Latitude", "Longitude", "Elevation (km)"]
    for i in range(3):
        axes[i].plot(samples[:, :, i], alpha=0.4)
        axes[i].set_ylabel(labels[i])
    axes[-1].set_xlabel("Step Number")
    plt.suptitle("MCMC Walker Convergence")
    plt.tight_layout()
    plt.savefig("mcmc_convergence.png")

    try:
        import corner
        fig = corner.corner(flat_samples, labels=labels, truths=[best_lat, best_lon, best_z])
        fig.savefig("mcmc_corner.png")
    except ImportError:
        print("Install `corner` to get corner plot.")

# === Output ===
true_lat = inv[0][0].latitude
true_lon = inv[0][0].longitude
true_elv = inv[0][0].elevation / 1000.0

print(f"Estimated station: lat={best_lat:.3f}, lon={best_lon:.3f}, elev={best_z:.2f} km")
print(f"True station:      lat={true_lat:.3f}, lon={true_lon:.3f}, elev={true_elv:.2f} km")

# === Misfit surface plot (lat-lon at best_z) ===
# print("Generating misfit surface plot...")

# lat_vals = np.linspace(LAT_RANGE[0], LAT_RANGE[1], 100)
# lon_vals = np.linspace(LON_RANGE[0], LON_RANGE[1], 100)
# misfit_surface = np.zeros((len(lat_vals), len(lon_vals)))

# with tqdm(total=len(lat_vals) * len(lon_vals), desc="Evaluating Surface") as pbar:
#     for i, lat in enumerate(lat_vals):
#         for j, lon in enumerate(lon_vals):
#             misfit_surface[i, j] = compute_misfit_for_location(lat, lon, best_z)
#             pbar.update(1)

# LAT, LON = np.meshgrid(lon_vals, lat_vals)

# plt.figure(figsize=(10, 6))
# cs = plt.contourf(LON, LAT, misfit_surface, levels=30, cmap='viridis')
# plt.colorbar(cs, label='Misfit (s²)')
# plt.plot(true_lon, true_lat, 'go', label='True Location')
# plt.plot(best_lon, best_lat, 'r*', markersize=12, label='Inverted Location')
# plt.plot(LON_CEN, LAT_CEN, 'c^', markersize=8, label='Initial Guess')
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.title(f"Misfit Surface at Depth {best_z:.2f} km")
# plt.legend()
# plt.tight_layout()
# plt.savefig("misfit_surface_map.png")
# plt.close()
# print("Saved: misfit_surface_map.png")

# === GMT Mercator Map ===
print("Plotting GMT Mercator map...")
lat_min_data = min(event_lats + [true_lat, best_lat]) - 2
lat_max_data = max(event_lats + [true_lat, best_lat]) + 2
lon_min_data = min(event_lons + [true_lon, best_lon]) - 2
lon_max_data = max(event_lons + [true_lon, best_lon]) + 2

lat_min_input, lat_max_input = LAT_RANGE
lon_min_input, lon_max_input = LON_RANGE

lat_min = min(lat_min_data, lat_min_input)
lat_max = max(lat_max_data, lat_max_input)
lon_min = min(lon_min_data, lon_min_input)
lon_max = max(lon_max_data, lon_max_input)

region = [lon_min, lon_max, lat_min, lat_max]


fig = pygmt.Figure()
fig.basemap(region=region, projection="M6i", frame=["af", f"+tStation Inversion: {STATION}"])
fig.grdimage(grid="@earth_relief_01m", region=region, projection="M6i", shading=True)

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
fig.plot(x=[best_lon], y=[best_lat], style="x0.4c", fill="red", pen="black", label="Inverted Station")
fig.plot(x=[LON_CEN], y=[LAT_CEN], style="t0.3c", fill="orange", pen="black", label="Initial Estimate")

for elat, elon in zip(event_lats, event_lons):
    fig.plot(data=[[elon, elat], [best_lon, best_lat]], pen="0.3p,gray,-")

# Misfit value annotation
misfit_km = np.sqrt((best_lat - true_lat)**2 + (best_lon - true_lon)**2 + (best_z - true_elv)**2)
fig.text(
    x=lon_min + 0.5, y=lat_min + 0.5,
    text=f"Misfit: {misfit_km:.2f}°",
    font="10p,Helvetica-Bold,black", justify="BL", offset="0.1c/0.1c")

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

# === Plot arrival picks ===
print("Plotting arrival picks...")
fig, ax = plt.subplots(figsize=(12, 6))
for ev_lat, ev_lon, t_obs, t0_evt in zip(event_lats, event_lons, observed_arrivals, origin_times):
    delta = (t_obs - t0_evt)
    ax.plot(delta, ev_lat, 'bo')

ax.set_xlabel("Predicted Arrival Time after Origin (s)")
ax.set_ylabel("Event Latitude")
ax.set_title("Picked P-wave Arrival Delays by Event")
plt.grid(True)
plt.tight_layout()
plt.savefig("arrival_picks.png")
print("Saved: arrival_picks.png")
