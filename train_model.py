import pandas as pd
import requests
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import joblib

SUPABASE_URL = "https://jawdhtalovhqoorwfrkt.supabase.co"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imphd2RodGFsb3ZocW9vcndmcmt0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjA4ODUzMjEsImV4cCI6MjA3NjQ2MTMyMX0.iKrhE_b3lL0CBEQFnTkFVbvK04aqrQ8eWQeloyyMJpg"
TABLE_URL = f"{SUPABASE_URL}/rest/v1/sensor_data"

headers = {"apikey": API_KEY, "Authorization": f"Bearer {API_KEY}"}

print("📡 Fetching data from Supabase...")
res = requests.get(TABLE_URL, headers=headers)
data = pd.DataFrame(res.json())
if data.empty:
    raise Exception("No sensor data found in Supabase!")

print(f"✅ Retrieved {len(data)} rows.")

X = data[["tds_value", "temperature", "water_level"]].fillna(0)
y = X["tds_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = ExtraTreesRegressor(n_estimators=500, max_depth=15, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, "savehydro_tds_model.pkl")
print("✅ Model retrained and saved as savehydro_tds_model.pkl")
