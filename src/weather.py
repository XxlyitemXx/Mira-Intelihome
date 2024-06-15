import requests


def get_weather(city="Nakhon Pathom", country="TH"):
    """
    Fetches weather information from OpenWeatherMap API.

    Args:
        city (str, optional): Name of the city. Defaults to "Nakhon Pathom".
        country (str, optional): Country code. Defaults to "TH".

    Returns:
        str: Weather information string, or an error message.
    """
    try:
        api_key = "YOUR_API_KEY_HERE"
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        complete_url = (
            base_url + "appid=" + api_key + "&q=" + city + "," + country
        )
        response = requests.get(complete_url)
        data = response.json()

        if data["cod"] != "404":
            weather = data["weather"][0]["description"]
            temperature = round(data["main"]["temp"] - 273.15, 1)
            return (
                f"The weather in {city} is {weather} with a temperature of {temperature}°C"
            )
        else:
            return "City Not Found"
    except Exception as e:
        print(f"Error in get_weather: {e}")
        return "An error occurred while fetching weather information."