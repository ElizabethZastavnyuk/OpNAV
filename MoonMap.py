import numpy as np
import matplotlib.pyplot as plt
import requests
import rasterio
import os

url = "https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif"
local_file = "moon_dem.tif"

if not os.path.exists(local_file):
    print("Downloading... (one time only)")
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    downloaded = 0
    with open(local_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024*1024):
            f.write(chunk)
            downloaded += len(chunk)
            mb = downloaded / (1024*1024)
            total_mb = total / (1024*1024)
            pct = (downloaded / total) * 100 if total else 0
            bar = ('█' * int(pct // 2)).ljust(50)
            print(f"\r  [{bar}] {pct:.1f}% — {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)
    print("\nDownload complete!")
else:
    print("File already downloaded, loading...")

print("Reading data at reduced resolution...")
with rasterio.open(local_file) as dataset:
    elevation = dataset.read(
        1,
        out_shape=(dataset.height // 16, dataset.width // 16)
    ).astype(np.float32)

elevation = elevation * 0.5
elevation[elevation < -20000] = np.nan

print(f"Grid shape: {elevation.shape}")
print(f"Elevation range: {np.nanmin(elevation):.0f} m to {np.nanmax(elevation):.0f} m")

print("Rendering map...")
fig, ax = plt.subplots(figsize=(16, 8))
img = ax.imshow(
    elevation,
    cmap='gist_earth',
    origin='upper',
    extent=[-180, 180, -90, 90],
    aspect='equal'
)
plt.colorbar(img, ax=ax, label='Elevation (m)', shrink=0.5)
ax.set_title('Full Moon Elevation Map — LOLA Global DEM 118m', fontsize=16)
ax.set_xlabel('Longitude (°)')
ax.set_ylabel('Latitude (°)')
ax.grid(True, alpha=0.3, color='white')
plt.tight_layout()
plt.savefig('moon_global.png', dpi=150)
plt.show()
print("Saved to moon_global.png")