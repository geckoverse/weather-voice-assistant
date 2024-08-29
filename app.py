from constants import (
    REPO_ID,
    WEATHER_URL,
    TEMPLATE1_FR_URL,
    TEMPLATE2_FR_URL,
    IMAGE_BASE_URL,
)
from dotenv import load_dotenv
import os
import subprocess
import speech_recognition as sr
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
import datetime as dt
import json
import requests
import streamlit as st
import pycountry
import matplotlib.pyplot as plt

# loading the env variables
load_dotenv()


def listen(language: str) -> str:
    recognizer = sr.Recognizer()
    with sr.Microphone() as micro:
        recognizer.adjust_for_ambient_noise(source=micro, duration=0.5)
        st.write("Listening...")
        audio = recognizer.listen(micro)

        try:
            text = recognizer.recognize_google(audio, language=language)
            return text
        except sr.UnknownValueError:
            return "Google Speech Recognition couldn't understand the audio"
        except sr.RequestError as e:
            return (
                f"Could not request results from Google Speech Recognition service; {e}"
            )


def load_llm(temperature: float, max_new_tokens: int) -> HuggingFaceEndpoint:
    hugging_face_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hugging_face_token
    llm = HuggingFaceEndpoint(
        repo_id=REPO_ID, temperature=temperature, max_new_tokens=max_new_tokens
    )
    return llm


def load_template(file_path: str) -> str:
    with open(file_path, "r") as file:
        return file.read()


def transform_json(file: str) -> dict:
    bracket_position = file.find("}")
    file = file[: bracket_position + 1]
    return json.loads(file)


def get_current_day() -> str:
    weekdays = {
        "Monday": "Lundi",
        "Tuesday": "Mardi",
        "Wednesday": "Mercredi",
        "Thursday": "Jeudi",
        "Friday": "Vendredi",
        "Saturday": "Samedi",
        "Sunday": "Dimanche",
    }
    return weekdays.get(dt.datetime.now().strftime("%A"))


def get_current_weather(city: str) -> dict:
    open_weather_map_token = os.getenv("OPEN_WEATHER_MAP_TOKEN")
    url = f"{WEATHER_URL}/data/2.5/weather?&q={city}&appid={open_weather_map_token}&lang=fr&units=metric"
    response = requests.get(url).json()
    return response


def get_forecast_weather(city: str) -> dict:
    open_weather_map_token = os.getenv("OPEN_WEATHER_MAP_TOKEN")
    url = f"{WEATHER_URL}/data/2.5/forecast?&q={city}&appid={open_weather_map_token}&lang=fr&units=metric"
    response = requests.get(url).json()
    return response


def round_hour(hour: int) -> str:
    hours = [0, 3, 6, 9, 12, 15, 18, 21]
    return str(min(hours, key=lambda x: abs(x - hour)))


def get_air_quality(latitude: float, longitude: float) -> int:
    open_weather_map_token = os.getenv("OPEN_WEATHER_MAP_TOKEN")
    pollution_url = f"{WEATHER_URL}/data/2.5/air_pollution?lat={latitude}&lon={longitude}&appid={open_weather_map_token}"
    air_pollution = requests.get(pollution_url).json()
    return air_pollution["list"][0]["main"]["aqi"]


def get_lat_lon_country(city: str) -> tuple[float, float, str]:
    open_weather_map_token = os.getenv("OPEN_WEATHER_MAP_TOKEN")
    coord_url = (
        f"{WEATHER_URL}/geo/1.0/direct?q={city}&limit=1&appid={open_weather_map_token}"
    )
    coord_json = requests.get(coord_url).json()
    latitude = coord_json[0]["lat"]
    longitude = coord_json[0]["lon"]
    country = coord_json[0]["country"]
    return latitude, longitude, country


def get_country_name(country_code: str) -> str:
    country = pycountry.countries.get(alpha_2=country_code)
    if country is None:
        return "Pays non trouvé"
    return country.name


def speak(
    text: str,
    voix: str,
    rate: str,
) -> None:
    command = command = ["say", "-v", voix, "-r", rate, text]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


def main():
    st.set_page_config(
        page_title="Weather Assistant", page_icon=":partly_sunny:", layout="centered"
    )

    col1, col2, col3 = st.columns([1, 6, 1])

    with col2:
        st.title("Weather Assistant")

        # voice parameters
        st.header("Voice Parameters")
        rate = st.slider("Rate", min_value=150, max_value=250, value=200)

        col4, col5 = st.columns(2)
        with col4:
            display_response = st.toggle("Display Response")
        with col5:
            display_forecast_graph = st.toggle("Display Forecast Graph")

        # vocal input
        st.header("Vocal Input")
        st.subheader("You can only see the weather for the next 5 days!")

        if st.button("Ask the weather"):
            question = listen("fr-FR")

            # load the llm
            llm = load_llm(temperature=0.1, max_new_tokens=128)

            # load the template
            template_content = load_template(TEMPLATE1_FR_URL)

            # question_meteo = "Quel temps fera-t-il demain à Lyon à cette heure ?"
            prompt = PromptTemplate.from_template(template=template_content)
            llm_chain = prompt | llm
            answer_weather = llm_chain.invoke(
                {
                    "weekday": get_current_day(),
                    "today": dt.datetime.now().strftime("%d/%m/%Y"),
                    "hour": dt.datetime.now().strftime("%H:%M"),
                    "question": question,
                }
            )

            json_weather = transform_json(answer_weather.strip())
            city = json_weather["ville"]

            if json_weather["heure"] != "None":
                hour = json_weather["heure"]
            else:
                hour = dt.datetime.now().strftime("%H")

            hour = round_hour(int(hour))

            if json_weather["date"] != "None":
                date = json_weather["date"]
                date = date.replace("/", "-")
                forecast_weather = get_forecast_weather(city)

                for dictionnary in forecast_weather["list"]:
                    dt_txt = dictionnary["dt_txt"]
                    if date in dt_txt and hour in dt_txt:
                        weather = dictionnary
                        sunrise = ""
                        sunset = ""
                        break
            else:
                date = dt.datetime.now().strftime("%d/%m/%Y")
                weather = get_current_weather(city)
                sunrise = dt.datetime.fromtimestamp(weather["sys"]["sunrise"]).strftime(
                    "%H:%M"
                )
                sunset = dt.datetime.fromtimestamp(weather["sys"]["sunset"]).strftime(
                    "%H:%M"
                )
            temperature_celcius = weather["main"]["temp"]
            temperature_celcius = round(temperature_celcius, 0)
            feels_like_celcius = weather["main"]["feels_like"]
            feels_like_celcius = round(feels_like_celcius, 0)
            humidity = weather["main"]["humidity"]
            wind_speed = weather["wind"]["speed"]
            description = weather["weather"][0]["description"]
            icon = weather["weather"][0]["icon"]
            image_url = f"{IMAGE_BASE_URL}/img/wn/{icon}@2x.png"

            st.header(f"Weather for {city} on {date} at {hour}h")
            st.image(image_url, caption=description)

            # get the latitude, longitude and country of the city
            latitude, longitude, country = get_lat_lon_country(city)

            # get the pollution with latitude and longitude
            air_quality_index = get_air_quality(latitude, longitude)

            col6, col7 = st.columns(2)
            with col6:
                st.subheader("🌡️ Température")
                st.write(f"{temperature_celcius}°C")

                st.subheader("🔥 Ressenti")
                st.write(f"{feels_like_celcius}°C")

                st.subheader("💧 Humidité")
                st.write(f"{humidity}%")

                if sunrise != "":
                    st.subheader("🌅 Levé du soleil")
                    st.write(sunrise)

            with col7:
                st.subheader("💨 Vitesse du vent")
                st.write(f"{wind_speed} m/s")

                st.subheader("🍃 Qualité de l'air")
                st.write(air_quality_index)

                st.subheader("🌍 Pays")
                country = get_country_name(country)
                st.write(country)

                if sunset != "":
                    st.subheader("🌇 Couché du soleil")
                    st.write(sunset)

            if display_forecast_graph:
                st.write("Forecast Graph for 5 days")

                dates = []
                temperatures = []
                for forecast in forecast_weather["list"]:
                    dates.append(forecast["dt_txt"])
                    temperatures.append(forecast["main"]["temp"])

                plt.plot(dates, temperatures, marker="o", linestyle="-")
                plt.title("Temperature Forecast 5 days")
                plt.xlabel("Date")
                plt.ylabel("Temperature (°C)")
                plt.xticks(dates[::2], rotation=45)
                plt.grid(True)
                plt.tight_layout()
                st.pyplot(plt)

            st.write("Prepation of Miss Weather's answer")

            # load the second template
            template_assistant = load_template(TEMPLATE2_FR_URL)
            prompt_assistant = PromptTemplate.from_template(template=template_assistant)
            llm_chain_assistant = prompt_assistant | llm
            answer_assistant = llm_chain_assistant.invoke(
                {
                    "weekday": get_current_day(),
                    "date": date,
                    "temperature_celcius": temperature_celcius,
                    "feels_like_celcius": feels_like_celcius,
                    "humidity": humidity,
                    "wind_speed": wind_speed,
                    "sunrise": sunrise,
                    "sunset": sunset,
                    "city": city,
                    "description": description,
                }
            )

            if display_response:
                st.write(answer_assistant)

            # Vocal answer
            speak(answer_assistant, voix="Thomas", rate=str(rate))
            st.write("Audio answer well generate")


if __name__ == "__main__":
    main()
