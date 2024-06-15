import speech_recognition as sr
from datetime import datetime
from gtts import gTTS
import playsound
import google.generativeai as genai

from weather import get_weather
from detector import detector


genai.configure(api_key="YOUR_API_KEY_HERE")


def text_to_speech(text, lang="en"):
    try:
        tts = gTTS(text=text, lang=lang)
        tts.save("output.mp3")
        playsound.playsound("output.mp3")
    except Exception as e:
        print(f"Error in text_to_speech: {e}")


def main():
    while True:
        recognizer = sr.Recognizer()
        mic = sr.Microphone()

        try:
            with mic as source:
                print("Listening...")
                audio = recognizer.listen(source)

            text = recognizer.recognize_google(audio, language="en-EN")
            print("You said:", text)

            if text.lower() in [
                "what time is it",
                "what's the time",
                "tell me the time",
            ]:
                now = datetime.now()
                current_time = now.strftime("%I:%M %p")
                text_to_speech(f"Right now is {current_time}")

            elif text.lower() in [
                "how many fingers on your frame",
                "how many fingers can you see",
                "count my fingers",
            ]:
                k = detector("finger")
                if k is not None:
                    r = f"I see {k} fingers in the frame."
                    print(r)
                    text_to_speech(r)
                else:
                    text_to_speech("Sorry, there was an error.")

            elif text.lower() in [
                "how is the weather",
                "what's the weather like",
                "tell me the weather",
            ]:
                info = get_weather()
                text_to_speech(info)

            elif text.lower() in [
                "how many people on your frame",
                "how many people can you see",
                "count the people",
            ]:
                f = detector("human")
                if f is not None:
                    text_to_speech(f"I see {f} people in the frame.")
                else:
                    text_to_speech("Sorry, there was an error.")

            else:
                generation_config = {
                    "temperature": 0.85,
                    "top_p": 0.95,
                    "top_k": 64,
                    "max_output_tokens": 8192,
                    "stop_sequences": ["service off", "system off", "end"],
                    "response_mime_type": "text/plain",
                }

                model = genai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    generation_config=generation_config,
                    system_instruction="", ## can't tell!
                )

                chat_session = model.start_chat(
                    history=[
                        ## can't tell!
                    ]
                )

                response = chat_session.send_message(text)
                print(response.text)
                text_to_speech(response.text)

        except sr.UnknownValueError:
            print("Could not understand audio")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except Exception as e:
            print("An error occurred:", e)


if __name__ == "__main__":
    main()