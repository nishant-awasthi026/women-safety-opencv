# alert_system.py
import pywhatkit as kit
import requests

class AlertSystem:
    def __init__(self, recipient_phone):
        self.recipient_phone = recipient_phone
        self.google_maps_api_key = "AIzaSyC6UBHOuZA6MnWlRa_PABe_Ag7hWZZIJ5c"

    def get_current_location(self):
        try:
            # Use IP-based geolocation API
            response = requests.get('https://ipinfo.io/json')
            data = response.json()

            # Extract coordinates
            loc = data.get('loc', '0,0').split(',')
            latitude = float(loc[0])
            longitude = float(loc[1])

            return latitude, longitude
        except requests.RequestException as e:
            print(f"Failed to get location: {e}")
            return 0, 0  # Return default coordinates if failed

    def get_map_url(self, latitude, longitude):
        # Generate a Google Maps URL for the given coordinates
        return (f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"
                f"&key={self.google_maps_api_key}")

    def send_alert(self, message):
        try:
            # Get current location coordinates
            lat, lon = self.get_current_location()

            # Generate Google Maps URL
            map_url = self.get_map_url(lat, lon)

            # Format the message to include location and coordinates
            full_message = (f"{message}\n\n"
                            f"Current Location:\n"
                            f"Latitude: {lat}\n"
                            f"Longitude: {lon}\n"
                            f"Map: {map_url}")

            # Send the message via WhatsApp
            kit.sendwhatmsg_instantly(self.recipient_phone, full_message)
            print("Alert sent successfully via WhatsApp!")
        except Exception as e:
            print(f"Failed to send alert: {e}")
