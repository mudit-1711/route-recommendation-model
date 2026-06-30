import pandas as pd
import numpy as np
import os
import pickle
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from datetime import datetime

def parse_bus_duration(duration_str):
    try:
        hours = int(re.search(r'(\d+)hrs', duration_str).group(1))
        mins = int(re.search(r'(\d+)mins', duration_str).group(1))
        return hours * 60 + mins
    except:
        return 0

def clean_city_name(city):
    if not isinstance(city, str):
        return city
    city = city.strip()
    c = city.lower()
    if c in ['banglore', 'bangalore', 'bengaluru', 'bengalore']:
        return 'Bangalore'
    if c in ['new delhi', 'delhi', 'new_delhi']:
        return 'Delhi'
    if c in ['mumbai', 'bombay']:
        return 'Mumbai'
    if c in ['chennai', 'madras']:
        return 'Chennai'
    if c in ['kolkata', 'calcutta']:
        return 'Kolkata'
    if c in ['hyderabad', 'hydrabad']:
        return 'Hyderabad'
    if c in ['pune', 'poona']:
        return 'Pune'
    return city.title()

def load_and_preprocess(data_dir="d:/far away data set/ml model"):
    print("Loading datasets...")
    
    # 1. Flights
    flights_df = pd.read_csv(os.path.join(data_dir, "finalairways.csv"))
    flights_df.dropna(inplace=True)
    flights_df['Source'] = flights_df['Source'].apply(clean_city_name)
    flights_df['Destination'] = flights_df['Destination'].apply(clean_city_name)
    flights_df['Transport_Type'] = 'Flight'
    # Flights: real duration estimated from typical flight time by city pair
    # Average commercial flight speed ~800 km/h. Distance is synthesized but duration
    # is set to reflect ACTUAL median flight times between Indian city pairs.
    FLIGHT_DURATION_MAP = {
        ('Delhi',     'Bangalore'):  150, ('Bangalore', 'Delhi'):     150,
        ('Delhi',     'Mumbai'):     120, ('Mumbai',    'Delhi'):     120,
        ('Delhi',     'Chennai'):    165, ('Chennai',   'Delhi'):     165,
        ('Delhi',     'Kolkata'):    135, ('Kolkata',   'Delhi'):     135,
        ('Delhi',     'Hyderabad'): 135, ('Hyderabad', 'Delhi'):     135,
        ('Delhi',     'Goa'):        165, ('Goa',       'Delhi'):     165,
        ('Mumbai',    'Bangalore'):  100, ('Bangalore', 'Mumbai'):    100,
        ('Mumbai',    'Chennai'):    105, ('Chennai',   'Mumbai'):    105,
        ('Mumbai',    'Kolkata'):    150, ('Kolkata',   'Mumbai'):    150,
        ('Mumbai',    'Hyderabad'):  90,  ('Hyderabad', 'Mumbai'):     90,
        ('Bangalore', 'Chennai'):    75,  ('Chennai',   'Bangalore'):  75,
        ('Bangalore', 'Kolkata'):   150,  ('Kolkata',   'Bangalore'): 150,
        ('Chennai',   'Kolkata'):   155,  ('Kolkata',   'Chennai'):   155,
    }
    def get_flight_duration(src, dst):
        return FLIGHT_DURATION_MAP.get((src, dst), 140)  # 140 min default

    flights_df['Distance'] = np.where(
        flights_df.apply(lambda r: (r['Source'], r['Destination']) in FLIGHT_DURATION_MAP, axis=1),
        flights_df.apply(lambda r: FLIGHT_DURATION_MAP.get((r['Source'], r['Destination']), 140) * 13, axis=1),
        np.random.uniform(1200, 1800, size=len(flights_df))
    )
    flights_df['Duration'] = flights_df.apply(
        lambda r: get_flight_duration(r['Source'], r['Destination']), axis=1
    ).astype(float)
    # Standardize column names
    flights_df = flights_df.rename(columns={'Airline': 'Mode', 'Date_of_Journey': 'Date'})
    flights_df = flights_df[['Transport_Type', 'Mode', 'Source', 'Destination', 'Date', 'Price', 'Distance', 'Duration']]
    
    # 2. Bangalore Roadways
    blr_cabs = pd.read_csv(os.path.join(data_dir, "finalbangloreroadways.csv"))
    blr_cabs.dropna(inplace=True)
    blr_cabs = blr_cabs[blr_cabs['Booking Value'] > 0]
    blr_cabs['Transport_Type'] = 'Cab'
    blr_cabs['Date'] = '24/03/2019' # Defaulting to a common date format for consistency
    blr_cabs['Duration'] = blr_cabs['Ride Distance'] * 4 # roughly 4 mins per km in city
    blr_cabs = blr_cabs.rename(columns={'Vehicle Type': 'Mode', 'Pickup Location': 'Source', 'Drop Location': 'Destination', 'Booking Value': 'Price', 'Ride Distance': 'Distance'})
    # Append 'Bangalore' to areas if we want to differentiate, but they are intra-city.
    blr_cabs['Source'] = blr_cabs['Source'] + ' (Bangalore)'
    blr_cabs['Destination'] = blr_cabs['Destination'] + ' (Bangalore)'
    blr_cabs = blr_cabs[['Transport_Type', 'Mode', 'Source', 'Destination', 'Date', 'Price', 'Distance', 'Duration']]
    
    # 3. Delhi Roadways
    del_cabs = pd.read_csv(os.path.join(data_dir, "finaldelhiroadways.csv"))
    del_cabs.dropna(inplace=True)
    del_cabs = del_cabs[del_cabs['Booking Value'] > 0]
    del_cabs['Transport_Type'] = 'Cab'
    del_cabs['Date'] = '24/03/2019'
    del_cabs['Duration'] = del_cabs['Ride Distance'] * 4
    del_cabs = del_cabs.rename(columns={'Vehicle Type': 'Mode', 'Pickup Location': 'Source', 'Drop Location': 'Destination', 'Booking Value': 'Price', 'Ride Distance': 'Distance'})
    del_cabs['Source'] = del_cabs['Source'] + ' (Delhi)'
    del_cabs['Destination'] = del_cabs['Destination'] + ' (Delhi)'
    del_cabs = del_cabs[['Transport_Type', 'Mode', 'Source', 'Destination', 'Date', 'Price', 'Distance', 'Duration']]
    
    # 4. Railways
    rail_df = pd.read_csv(os.path.join(data_dir, "finalrailways.csv"))
    rail_df.dropna(inplace=True)
    rail_df['Transport_Type'] = 'Train'
    rail_df['Date'] = pd.to_datetime(rail_df['timeStamp']).dt.strftime('%d/%m/%Y')
    rail_df = rail_df.rename(columns={'trainNumber': 'Mode', 'fromStnCode': 'Source', 'toStnCode': 'Destination', 'totalFare': 'Price', 'distance': 'Distance', 'duration': 'Duration'})
    rail_df['Mode'] = rail_df['Mode'].astype(str)
    # Simple mapping for station codes to city names if they match Delhi/Bangalore (optional, but keep it simple)
    # E.g. NDLS -> Delhi, SBC -> Bangalore
    # Comprehensive Indian railway station code to city name mapping
    stn_map = {
        # Delhi / NCR
        'NDLS': 'Delhi', 'NZM': 'Delhi', 'SZM': 'Delhi', 'DLI': 'Delhi',
        'DSA': 'Delhi', 'DEE': 'Delhi', 'ANVT': 'Delhi', 'HZM': 'Delhi',
        # Bangalore
        'SBC': 'Bangalore', 'YPR': 'Bangalore', 'BNC': 'Bangalore', 'KJM': 'Bangalore',
        'BAND': 'Bangalore', 'BYPL': 'Bangalore', 'BNCE': 'Bangalore',
        # Mumbai
        'CSTM': 'Mumbai', 'BCT': 'Mumbai', 'LTT': 'Mumbai', 'DR': 'Mumbai',
        'TNA': 'Mumbai', 'KYN': 'Mumbai', 'BSR': 'Mumbai', 'VR': 'Mumbai',
        'VSVD': 'Mumbai', 'DRD': 'Mumbai', 'BVI': 'Mumbai', 'MMR': 'Mumbai',
        # Chennai
        'MAS': 'Chennai', 'MS': 'Chennai', 'MSC': 'Chennai', 'PRES': 'Chennai',
        'TBM': 'Chennai', 'PER': 'Chennai', 'AVD': 'Chennai',
        # Kolkata
        'HWH': 'Kolkata', 'SDAH': 'Kolkata', 'KOL': 'Kolkata', 'KOAA': 'Kolkata',
        'BDC': 'Kolkata', 'GRAE': 'Kolkata',
        # Hyderabad
        'SC': 'Hyderabad', 'HYB': 'Hyderabad', 'KCG': 'Hyderabad', 'LPI': 'Hyderabad',
        'BMT': 'Hyderabad', 'SNF': 'Hyderabad',
        # Ahmedabad
        'ADI': 'Ahmedabad', 'MAN': 'Ahmedabad', 'SBT': 'Ahmedabad',
        # Pune
        'PUNE': 'Pune', 'PUN': 'Pune', 'PUNE': 'Pune', 'DD': 'Pune',
        # Jaipur
        'JP': 'Jaipur', 'JPR': 'Jaipur',
        # Lucknow
        'LKO': 'Lucknow', 'LJN': 'Lucknow',
        # Agra
        'AGC': 'Agra', 'AF': 'Agra',
        # Varanasi
        'BSB': 'Varanasi', 'MUV': 'Varanasi',
        # Goa
        'MAO': 'Goa', 'LD': 'Goa', 'THVM': 'Goa',
        # Chandigarh
        'CDG': 'Chandigarh', 'UMB': 'Chandigarh',
        # Amritsar
        'ASR': 'Amritsar',
        # Patna
        'PNBE': 'Patna', 'DNR': 'Patna',
        # Bhopal
        'BPL': 'Bhopal', 'HBJ': 'Bhopal',
        # Nagpur
        'NGP': 'Nagpur',
        # Surat
        'ST': 'Surat',
        # Kochi
        'ERS': 'Kochi', 'AWY': 'Kochi', 'ALLP': 'Kochi',
        # Coimbatore
        'CBE': 'Coimbatore',
        # Indore
        'INDB': 'Indore',
        # Visakhapatnam
        'VSKP': 'Visakhapatnam',
    }
    rail_df['Source'] = rail_df['Source'].map(lambda x: stn_map.get(x, x))
    rail_df['Destination'] = rail_df['Destination'].map(lambda x: stn_map.get(x, x))
    rail_df = rail_df[['Transport_Type', 'Mode', 'Source', 'Destination', 'Date', 'Price', 'Distance', 'Duration']]
    
    # 5. Intercity Buses
    bus_df = pd.read_csv(os.path.join(data_dir, "intercitybuses.csv"))
    bus_df.dropna(inplace=True)
    bus_df['Transport_Type'] = 'Bus'
    bus_df['Mode'] = 'Intercity Bus'
    bus_df['Date'] = '24/03/2019'
    bus_df['Duration'] = bus_df['Travel Duration'].apply(parse_bus_duration)
    bus_df = bus_df.rename(columns={'price': 'Price', 'distance': 'Distance'})
    bus_df['Source'] = bus_df['Source'].apply(clean_city_name)
    bus_df['Destination'] = bus_df['Destination'].apply(clean_city_name)
    bus_df = bus_df[['Transport_Type', 'Mode', 'Source', 'Destination', 'Date', 'Price', 'Distance', 'Duration']]
    
    # Merge all
    df = pd.concat([flights_df, blr_cabs, del_cabs, rail_df, bus_df], ignore_index=True)
    
    print(f"Total merged records: {len(df)}")
    
    # Date Features
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    # For dates that failed to parse, forward fill or use a default
    df['Date'] = df['Date'].fillna(pd.to_datetime('24/03/2019', format='%d/%m/%Y'))
    
    df['Day'] = df['Date'].dt.day
    df['Month'] = df['Date'].dt.month
    df['Weekday'] = df['Date'].dt.weekday
    df['Is_Weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Categorical Encoding
    print("Encoding categorical features...")
    cat_cols = ['Transport_Type', 'Mode', 'Source', 'Destination']
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col+'_encoded'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        
    # Numerical Scaling
    print("Scaling numerical features...")
    num_cols = ['Distance', 'Duration', 'Price']
    scaler = MinMaxScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    
    # Save artifacts
    artifacts_dir = os.path.join(data_dir, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with open(os.path.join(artifacts_dir, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
        
    with open(os.path.join(artifacts_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
        
    df.to_csv(os.path.join(artifacts_dir, 'processed_data.csv'), index=False)

    # Save REAL (unscaled) stats for the recommendation engine
    # raw_df still has original prices/durations (before MinMaxScaler was applied to df)
    raw_df = pd.concat([flights_df, blr_cabs, del_cabs, rail_df, bus_df], ignore_index=True)
    raw_df = raw_df[raw_df['Price'] > 0]
    raw_df = raw_df[raw_df['Duration'] > 0]

    # 1. Route-level lookup: Transport_Type + Source + Destination -> median price & duration
    route_lookup = raw_df.groupby(['Transport_Type', 'Source', 'Destination']).agg(
        median_price=('Price', 'median'),
        avg_price=('Price', 'mean'),
        median_duration=('Duration', 'median'),
        avg_duration=('Duration', 'mean'),
        median_distance=('Distance', 'median'),
        count=('Price', 'count'),
    ).reset_index()

    # Mirror A->B entries as B->A so reverse routes always get a dataset match
    # (flights dataset only had Bangalore->Delhi but not Delhi->Bangalore)
    reverse = route_lookup.copy()
    reverse['Source'], reverse['Destination'] = route_lookup['Destination'], route_lookup['Source']
    route_lookup = pd.concat([route_lookup, reverse], ignore_index=True)
    route_lookup = route_lookup.drop_duplicates(subset=['Transport_Type','Source','Destination'], keep='first')

    route_lookup.to_csv(os.path.join(artifacts_dir, 'route_lookup.csv'), index=False)
    print(f"Route-level lookup saved: {len(route_lookup)} unique routes (including mirrored).")

    # 2. Transport-type fallback stats (used when no exact route match exists)
    transport_stats = raw_df.groupby('Transport_Type').agg(
        avg_price=('Price', 'mean'),
        median_price=('Price', 'median'),
        min_price=('Price', 'min'),
        avg_duration=('Duration', 'mean'),
        median_duration=('Duration', 'median'),
        min_duration=('Duration', 'min'),
        avg_distance=('Distance', 'mean'),
    ).reset_index()
    transport_stats.to_csv(os.path.join(artifacts_dir, 'transport_stats.csv'), index=False)
    print("Transport-type fallback stats saved.")
    print(transport_stats.to_string())

    # 3. City-specific cab pricing stats from actual Uber/Ola dataset
    # These give the REAL per-km price and speed for each city
    cab_raw = raw_df[raw_df['Transport_Type'] == 'Cab'].copy()
    cab_raw['City'] = cab_raw['Source'].apply(
        lambda x: 'Delhi' if '(Delhi)' in str(x) else ('Bangalore' if '(Bangalore)' in str(x) else None)
    )
    cab_raw = cab_raw.dropna(subset=['City'])
    cab_raw = cab_raw[cab_raw['Distance'] > 0.5]

    city_cab_stats = cab_raw.groupby('City').apply(lambda g: pd.Series({
        'price_per_km': (g['Price'] / g['Distance']).median(),
        'min_per_km':   (g['Duration'] / g['Distance']).median(),
        'median_price':    g['Price'].median(),
        'median_distance': g['Distance'].median(),
        'median_duration': g['Duration'].median(),
    })).reset_index()
    city_cab_stats.to_csv(os.path.join(artifacts_dir, 'city_cab_stats.csv'), index=False)
    print("City cab stats saved (actual dataset per-km rates):")
    print(city_cab_stats.to_string())
    print("Preprocessing complete. Data and artifacts saved in", artifacts_dir)


if __name__ == "__main__":
    load_and_preprocess()
