import os
import gzip
import json
import requests
from datetime import timedelta, datetime

# -----------------------------
# CONFIGURATION
# -----------------------------
DATE = "2025/10/01"  # yyyy/mm/dd
OUT_DIR = "./adsb_data"
NUM_FILES = 100  # number of snapshots to download
REGION_BOUNDS = {
    "lamin": 6.0,   # latitude min
    "lamax": 36.0,  # latitude max
    "lomin": 68.0,  # longitude min
    "lomax": 98.0   # longitude max
}

# Base URL pattern (use sample date)
BASE_URL = f"https://samples.adsbexchange.com/readsb-hist/{DATE}/"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

def generate_filenames(num_files: int):
    """
    Generate realistic file names like 000000Z.json.gz, 000005Z.json.gz, etc.
    ADS-B Exchange archives files about every 5 seconds.
    """
    filenames = []
    start_time = datetime.strptime("000000", "%H%M%S")
    for i in range(num_files):
        t = (start_time + timedelta(seconds=i * 5)).strftime("%H%M%SZ")
        filenames.append(f"{t}.json.gz")
    return filenames


def download_file(filename: str):
    """Download one compressed ADS-B sample file."""
    url = BASE_URL + filename
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            print(f"❌ {filename}: HTTP {resp.status_code}")
            return None
        
        # Check if we got actual content
        if not resp.content:
            print(f"⚠️  {filename}: Empty response")
            return None
            
        print(f"⬇️  Downloaded {filename} ({len(resp.content)} bytes)")
        return resp.content
    except requests.exceptions.Timeout:
        print(f"⏰ Timeout downloading {filename}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"🔌 Connection error downloading {filename}")
        return None
    except Exception as e:
        print(f"⚠️  Error downloading {filename}: {e}")
        return None


def is_gzipped(content):
    """Check if content is gzipped by looking at magic bytes."""
    return content.startswith(b'\x1f\x8b')


def filter_by_region(states, bounds):
    """Filter aircraft states within specified lat/lon bounds."""
    filtered = []
    for st in states:
        if st[6] is None or st[5] is None:
            continue
        lat, lon = st[6], st[5]
        if bounds["lamin"] <= lat <= bounds["lamax"] and bounds["lomin"] <= lon <= bounds["lomax"]:
            filtered.append(st)
    return filtered


def process_and_save(content, filename):
    """Decompress, filter, and save JSON."""
    try:
        # Check if content is actually gzipped using magic bytes
        if is_gzipped(content):
            try:
                data = gzip.decompress(content)
                print(f"🗜️  Decompressed gzipped file: {filename}")
            except gzip.BadGzipFile as e:
                print(f"⚠️  {filename}: Corrupted gzip file - {e}")
                return
        else:
            # Plain JSON content
            data = content
            print(f"📄 Processing plain JSON: {filename}")
        
        # Parse JSON
        parsed = json.loads(data)
        
        # Filter by region if states exist
        if "states" in parsed:
            original_count = len(parsed["states"])
            parsed["states"] = filter_by_region(parsed["states"], REGION_BOUNDS)
            filtered_count = len(parsed["states"])
            print(f"🔍 Filtered {original_count} -> {filtered_count} flights")

        # Save to file
        os.makedirs(OUT_DIR, exist_ok=True)
        out_path = os.path.join(OUT_DIR, filename.replace(".gz", ""))
        with open(out_path, "w") as f:
            json.dump(parsed, f, indent=2)
        print(f"✅ Saved {out_path} ({len(parsed.get('states', []))} flights)")
    except json.JSONDecodeError as e:
        print(f"⚠️  {filename}: Invalid JSON - {e}")
    except Exception as e:
        print(f"⚠️  Failed to process {filename}: {e}")


# -----------------------------
# MAIN SCRIPT
# -----------------------------
def main():
    filenames = generate_filenames(NUM_FILES)
    print(f"🚀 Preparing to download {len(filenames)} files from {BASE_URL}")
    print(f"📁 Output directory: {OUT_DIR}")
    print(f"🌍 Region bounds: {REGION_BOUNDS}")
    print("-" * 50)
    
    successful = 0
    failed = 0
    
    for i, fname in enumerate(filenames, 1):
        print(f"\n[{i}/{len(filenames)}] Processing {fname}")
        content = download_file(fname)
        if content:
            process_and_save(content, fname)
            successful += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Summary: {successful} successful, {failed} failed")
    print(f"📁 Files saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
