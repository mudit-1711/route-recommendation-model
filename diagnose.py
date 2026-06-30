import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os

DATA_DIR = "d:/far away data set/ml model"
rl = pd.read_csv(f"{DATA_DIR}/artifacts/route_lookup.csv")

# 1. Check what cab routes exist for Delhi/Bangalore area pairs
print("=== CAB ROUTES FROM DATASET (sample, Delhi) ===")
dcabs = rl[(rl['Transport_Type']=='Cab') & rl['Source'].str.contains('Delhi')]
print(dcabs[['Source','Destination','median_price','median_duration','median_distance']].head(10).to_string())

print("\n=== CAB ROUTES FROM DATASET (sample, Bangalore) ===")
bcabs = rl[(rl['Transport_Type']=='Cab') & rl['Source'].str.contains('Bangalore')]
print(bcabs[['Source','Destination','median_price','median_duration','median_distance']].head(10).to_string())

# 2. Per-km rate from raw cab datasets
print("\n=== DELHI CAB DATASET PER-KM STATS ===")
del_raw = pd.read_csv(f"{DATA_DIR}/finaldelhiroadways.csv")
del_raw = del_raw[del_raw['Booking Value'] > 0]
del_raw = del_raw[del_raw['Ride Distance'] > 0]
del_raw['price_per_km'] = del_raw['Booking Value'] / del_raw['Ride Distance']
del_raw['min_per_km'] = (del_raw['Ride Distance'] * 4) / del_raw['Ride Distance']  # 4 min/km assumed
print(f"  Records: {len(del_raw)}")
print(f"  Median price_per_km: Rs.{del_raw['price_per_km'].median():.2f}")
print(f"  Median price for 25km: Rs.{del_raw['price_per_km'].median()*25:.0f}")
print(f"  Median price for 12km: Rs.{del_raw['price_per_km'].median()*12:.0f}")
print(f"  Median Booking Value: Rs.{del_raw['Booking Value'].median():.0f}")
print(f"  Median Ride Distance: {del_raw['Ride Distance'].median():.2f} km")

print("\n=== BANGALORE CAB DATASET PER-KM STATS ===")
blr_raw = pd.read_csv(f"{DATA_DIR}/finalbangloreroadways.csv")
blr_raw = blr_raw[blr_raw['Booking Value'] > 0]
blr_raw = blr_raw[blr_raw['Ride Distance'] > 0]
blr_raw['price_per_km'] = blr_raw['Booking Value'] / blr_raw['Ride Distance']
print(f"  Records: {len(blr_raw)}")
print(f"  Median price_per_km: Rs.{blr_raw['price_per_km'].median():.2f}")
print(f"  Median price for 25km: Rs.{blr_raw['price_per_km'].median()*25:.0f}")
print(f"  Median price for 12km: Rs.{blr_raw['price_per_km'].median()*12:.0f}")
print(f"  Median Booking Value: Rs.{blr_raw['Booking Value'].median():.0f}")
print(f"  Median Ride Distance: {blr_raw['Ride Distance'].median():.2f} km")

# 3. Check flight + train for Delhi<->Bangalore
print("\n=== DELHI <-> BANGALORE: ALL TRANSPORT MODES ===")
for mode in ['Flight','Train','Bus']:
    m = rl[(rl['Transport_Type']==mode) &
           (rl['Source'].str.lower()=='delhi') &
           (rl['Destination'].str.lower()=='bangalore')]
    if not m.empty:
        row = m.iloc[0]
        print(f"  {mode}: price=Rs.{row['median_price']:.0f}, duration={row['median_duration']:.0f}min ({row['median_duration']/60:.1f}h), count={int(row['count'])}")
    else:
        print(f"  {mode}: NO MATCH in route_lookup")

# 4. Simulate recommend for all preferences Delhi->Bangalore
print("\n=== SIMULATE ALL PREFERENCES: Delhi->Bangalore ===")
from recommend import RouteRecommender
r = RouteRecommender()
for pref in ['Cheapest','Fastest','Economical','Balanced','Luxury','Comfort optimized']:
    res = r.get_best_route('Delhi','Bangalore','Rohini','Whitefield','2026-06-12', pref)
    mode = next((l['Transport_Type'] for l in res['Route Details'] if l['Transport_Type'] != 'Cab'), 'Cab')
    print(f"  {pref:<22} {mode:<10} {res['Total Cost']:>8} {res['Estimated Time']:>10}")

print("\n=== SIMULATE ALL PREFERENCES: Bangalore->Delhi ===")
for pref in ['Cheapest','Fastest','Economical','Balanced','Luxury','Comfort optimized']:
    res = r.get_best_route('Bangalore','Delhi','Whitefield','Rohini','2026-06-12', pref)
    mode = next((l['Transport_Type'] for l in res['Route Details'] if l['Transport_Type'] != 'Cab'), 'Cab')
    print(f"  {pref:<22} {mode:<10} {res['Total Cost']:>8} {res['Estimated Time']:>10}")
