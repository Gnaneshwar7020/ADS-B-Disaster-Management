import json
import os
from typing import List, Dict
from datetime import datetime, timedelta
import random


def generate_synthetic_adsb_data(num_records: int = 500) -> List[Dict]:
    """Generate synthetic ADS-B dataset if it doesn't exist"""
    
    airlines = ["AA", "UA", "DL", "SW", "BA", "LH", "AF", "KL", "SQ", "EK"]
    aircraft_types = ["B787", "B777", "A380", "A350", "B737", "A320", "CRJ", "E190"]
    airports = ["JFK", "LAX", "ORD", "DFW", "ATL", "LHR", "CDG", "NRT", "SIN", "DXB"]
    
    records = []
    base_time = datetime.now()
    
    for i in range(num_records):
        record = {
            "flight_id": f"{random.choice(airlines)}{random.randint(1000, 9999)}",
            "icao_address": f"{random.randint(0xa0, 0xf0):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}",
            "callsign": f"{random.choice(airlines)}FL{i:04d}",
            "aircraft_type": random.choice(aircraft_types),
            "latitude": round(random.uniform(-90, 90), 4),
            "longitude": round(random.uniform(-180, 180), 4),
            "altitude_ft": random.randint(1000, 45000),
            "ground_speed_knots": random.randint(200, 500),
            "track_degrees": random.randint(0, 359),
            "vertical_rate_fpm": random.randint(-2000, 2000),
            "departure_airport": random.choice(airports),
            "destination_airport": random.choice(airports),
            "timestamp": (base_time - timedelta(seconds=random.randint(0, 86400))).isoformat(),
            "emergency_status": random.choice(["none", "general", "medical", "fuel"])
        }
        records.append(record)
    
    return records


def load_adsb_data(data_path: str = "data/adsb_synthetic.json") -> List[Dict]:
    """Load ADS-B data from JSON file, generate if not exists"""
    
    os.makedirs(os.path.dirname(data_path) if os.path.dirname(data_path) else ".", exist_ok=True)
    
    if not os.path.exists(data_path):
        print(f"Generating synthetic ADS-B data at {data_path}...")
        data = generate_synthetic_adsb_data(500)
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Generated {len(data)} ADS-B records")
    else:
        with open(data_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} ADS-B records from {data_path}")
    
    return data


def preprocess_adsb_data(records: List[Dict]) -> List[str]:
    """Convert flight records to searchable text format"""
    
    documents = []
    for record in records:
        doc = f"""
        Flight Information:
        - Flight ID: {record.get('flight_id', 'N/A')}
        - Callsign: {record.get('callsign', 'N/A')}
        - Aircraft Type: {record.get('aircraft_type', 'N/A')}
        - ICAO Address: {record.get('icao_address', 'N/A')}
        
        Current Position:
        - Latitude: {record.get('latitude', 'N/A')}
        - Longitude: {record.get('longitude', 'N/A')}
        - Altitude: {record.get('altitude_ft', 'N/A')} feet
        - Ground Speed: {record.get('ground_speed_knots', 'N/A')} knots
        - Track: {record.get('track_degrees', 'N/A')} degrees
        - Vertical Rate: {record.get('vertical_rate_fpm', 'N/A')} fpm
        
        Route:
        - Departure: {record.get('departure_airport', 'N/A')}
        - Destination: {record.get('destination_airport', 'N/A')}
        
        Status:
        - Emergency Status: {record.get('emergency_status', 'none')}
        - Timestamp: {record.get('timestamp', 'N/A')}
        """
        documents.append(doc.strip())
    
    return documents