import os
import subprocess
from pathlib import Path
from obspy import UTCDateTime, read, Stream
from obspy.clients.fdsn import Client
from obspy.core.event import Catalog, Pick
import pandas as pd

# === Parameters ===
NETWORK = "IU"
STATION = "ANMO"
LOCATION = "*"
CHANNELS = "BH?"  # Adjust as needed
STARTTIME = UTCDateTime("2025-07-14T00:00:00")
ENDTIME = STARTTIME + 86400  # One day of data

DATA_DIR = Path("data") / f"{NETWORK}.{STATION}"
PICKS_DIR = Path("picks")
PHASENET_MODEL = "phasenet/model/190703-214543"

# === Fetch Data ===
client = Client("IRIS")
print(f"Downloading data for {NETWORK}.{STATION} from {STARTTIME} to {ENDTIME}...")

try:
    st = client.get_waveforms(NETWORK, STATION, LOCATION, CHANNELS, STARTTIME, ENDTIME)
except Exception as e:
    print(f"Download failed: {e}")
    exit(1)

st.merge(method=1, fill_value="interpolate")
print(f"Downloaded {len(st)} traces")

# === Save Data for PhaseNet ===
DATA_DIR.mkdir(parents=True, exist_ok=True)
for tr in st:
    timestamp = STARTTIME.strftime('%Y%m%dT%H%M%SZ')
    fname = f"{timestamp}_{tr.id}.mseed"
    tr.write(DATA_DIR / fname, format="MSEED")
print(f"Saved traces to {DATA_DIR}")

# === Run PhaseNet ===
PICKS_DIR.mkdir(parents=True, exist_ok=True)
cmd = [
    "phasenet",
    "detect",
    "--model", PHASENET_MODEL,
    "--data_dir", str(DATA_DIR),
    "--output_dir", str(PICKS_DIR)
]

print(f"Running PhaseNet: {' '.join(cmd)}")
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"PhaseNet failed: {e}")
    exit(1)

print(f"PhaseNet detection complete. Picks saved to {PICKS_DIR}")

# === Load Picks into ObsPy Catalog ===
picks_file = PICKS_DIR / f"{NETWORK}.{STATION}.csv"
if not picks_file.exists():
    print(f"No picks found for {NETWORK}.{STATION}")
    exit(0)

df = pd.read_csv(picks_file)
cat = Catalog()

for _, row in df.iterrows():
    try:
        pick = Pick(
            time=UTCDateTime(row["time"]),
            phase_hint=row["phase"],
            method_id="smi:local/PhaseNet"
        )
        cat.picks.append(pick)
    except Exception as e:
        print(f"Error parsing pick: {e}")

print(f"Loaded {len(cat.picks)} picks into ObsPy Catalog")
