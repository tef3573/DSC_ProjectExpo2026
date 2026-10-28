import pandas as pd
from pathlib import Path

# Setup directories
delisted_dir = Path("data/delisted")
delisted_dir.mkdir(parents=True, exist_ok=True)

# Load raw JCP CSV
df_raw = pd.read_csv("JCP.csv")

# Clean whitespace and fix the clipped July 27 date
df_raw["Date"] = df_raw["Date"].astype(str).str.strip().replace({"ul 27 2016": "Jul 27 2016"})
df_raw["Date"] = pd.to_datetime(df_raw["Date"], errors="coerce")

# Clean Volume (remove commas, cast to float)
df_raw["Volume"] = df_raw["Volume"].astype(str).str.replace(",", "").astype(float)

# Drop market holidays (Volume == 0) and sort chronologically
df_clean = df_raw[df_raw["Volume"] > 0].dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)

# Export standard price columns
output = pd.DataFrame({
    "Date": df_clean["Date"].dt.strftime("%Y-%m-%d"),
    "Close": df_clean["Close"].round(3),
    "Adj Close": df_clean["Close"].round(3),
    "Volume": df_clean["Volume"].astype(int)
})

output.to_csv(delisted_dir / "JCP.csv", index=False)
print(f"Successfully saved {len(output)} trading sessions to data/delisted/JCP.csv")