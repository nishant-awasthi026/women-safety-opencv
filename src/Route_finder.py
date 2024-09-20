import requests
import json

ORS_API_KEY = '5b3ce3597851110001cf62487ea2b9edc1b443f9bac4205b634ca8e2'

def create_route(start_location, end_location, profile='driving-car'):
    """Creates a route using OpenRouteService API."""
    route_url = f"https://api.openrouteservice.org/v2/directions/{profile}?api_key={ORS_API_KEY}&start={start_location}&end={end_location}"
    response = requests.get(route_url)
    route_data = response.json()

    print("API Response:", json.dumps(route_data, indent=2))

    if 'routes' in route_data and len(route_data['routes']) > 0:
        route_info = route_data['routes'][0]['geometry']['coordinates']
        duration = route_data['routes'][0]['summary']['duration']
        distance = route_data['routes'][0]['summary']['distance']

        return {
            'route': route_info,
            'duration': duration,
            'distance': distance
        }
    else:
        raise Exception("Error fetching route data: No routes found or route data is empty")
