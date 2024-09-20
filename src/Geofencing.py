from geopy.distance import great_circle

SAFE_ZONE_CENTER = (77.3300, 77.3300)
SAFE_ZONE_RADIUS = 1000  

def is_within_safe_zone(current_location):
    """Check if the current location is within the safe zone."""
    distance = great_circle(SAFE_ZONE_CENTER, current_location).meters
    return distance <= SAFE_ZONE_RADIUS

def check_geofence(latitude, longitude):
    # Example geofencing check
    safe_zone = {'latitude': 40.7128, 'longitude': -74.0060}
    threshold = 0.1  # Example threshold for geofencing
    distance = ((latitude - safe_zone['latitude'])**2 + (longitude - safe_zone['longitude'])**2)**0.5
    if distance < threshold:
        return "Inside safe zone"
    else:
        return "Outside safe zone"

def main():
    current_lat = 28.5800
    current_lon = 77.3300

    try:
        check_geofence(current_lat, current_lon)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
