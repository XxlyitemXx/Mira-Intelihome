import requests
import json

def load_config():
    try:
        with open('config.json', 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("Configuration file not found. Please ensure 'config.json' exists.")
        exit(1)
    except json.JSONDecodeError:
        print("Error reading 'config.json'. Please ensure it contains valid JSON.")
        exit(1)

def get_weather_data(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = f"{base_url}q={city}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(complete_url)
        response.raise_for_status()  # Raise an HTTPError for bad responses
        data = response.json()

        if data["cod"] != 200:
            print(f"Error fetching data: {data['message']}")
            return None
        
        main = data.get("main", {})
        wind = data.get("wind", {})
        weather = data.get("weather", [{}])[0]
        rain = data.get("rain", {}).get("1h", 0)
        
        weather_data = {
            "temperature": main.get("temp"),
            "humidity": main.get("humidity"),
            "wind_speed": wind.get("speed"),
            "weather_description": weather.get("description", "No description"),
            "rain_rate": rain
        }
        return weather_data
    except requests.RequestException as e:
        print(f"Error making request to OpenWeatherMap API: {e}")
        return None

def display_weather_data(weather_data):
    if weather_data:
        print(f"Temperature: {weather_data['temperature']}°C")
        print(f"Humidity: {weather_data['humidity']}%")
        print(f"Wind Speed: {weather_data['wind_speed']} m/s")
        print(f"Weather Description: {weather_data['weather_description']}")
        print(f"Rain Rate: {weather_data['rain_rate']} mm/h")
    else:
        print("No weather data available.")

def main():
    config = load_config()
    api_key = config.get("api_key")

    if not api_key:
        print("API key not found in configuration file.")
        exit(1)

    city = input("Enter city name: ")

    weather_data = get_weather_data(city, api_key)
    display_weather_data(weather_data)

if __name__ == "__main__":
    main()
