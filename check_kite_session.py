import os
import json

SESSION_PATH = os.path.expanduser("~/.kite_session.json")

if not os.path.exists(SESSION_PATH):
    print(f"Session file not found: {SESSION_PATH}")
    exit(1)

with open(SESSION_PATH, "r", encoding="utf8") as f:
    data = json.load(f)

print("Session file contents:")
print(json.dumps(data, indent=2))

if 'access_token' in data and data['access_token']:
    print("Access token found and appears valid.")
else:
    print("No access_token found or token is empty/expired. Please re-run kite.py and complete login.")
