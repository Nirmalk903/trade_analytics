import requests
import pandas as pd
import os
import time

# Set up download directory
download_dir = os.path.abspath("Most_Active_Underlying")
os.makedirs(download_dir, exist_ok=True)

url = "https://www.nseindia.com/api/live-analysis-oi-spurts-underlyings"
headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/"
}

session = requests.Session()
session.headers.update(headers)

# Get the NSE homepage first to set cookies
try:
    home = session.get("https://www.nseindia.com", timeout=10)
    print("Homepage status:", home.status_code)
except Exception as e:
    print("Error connecting to NSE homepage:", e)
    exit()

time.sleep(3)  # Wait for cookies to be set

# Now get the data
try:
    response = session.get(url, timeout=10)
    print("API status:", response.status_code)
except Exception as e:
    print("Error connecting to API:", e)
    exit()

if response.status_code != 200:
    print("Failed to fetch data. Status code:", response.status_code)
    print("Response text:", response.text[:500])
    exit()

try:
    data = response.json()
except Exception as e:
    print("Error parsing JSON:", e)
    print("Raw response text:", response.text[:500])
    exit()

if not isinstance(data, dict):
    print("Unexpected data format received from API.")
    print("Raw response text:", response.text[:500])
    exit()

print("API JSON keys:", data.keys())
records = data.get('data', [])
if not records:
    print("No data found in API response.")
    print("Full API response:", data)
    exit()

df = pd.DataFrame(records)
csv_path = os.path.join(download_dir, "oi_spurts_underlyings.csv")
df.to_csv(csv_path, index=False)

print("Data successfully downloaded and saved to:", csv_path)