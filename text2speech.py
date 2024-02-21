from gtts import gTTS
import os

from playsound import playsound

# Definir el texto y el idioma
texto = "Hola, este es un ejemplo de texto a voz en español."
idioma = 'es'


def text2speech(text, lang="en"):
    # Crear un objeto gTTS
    tts = gTTS(text=text, lang=lang, slow=False)

    # Guardar el archivo de audio
    tts.save("answer.mp3")

    # Opcional: Reproducir el archivo de audio
    playsound("answer.mp3")

    os.system("rm answer.mp3")


if __name__ == "__main__":
    # Definir el texto y el idioma
    texto = "Hola, este es un ejemplo de texto a voz en español."
    idioma = 'es'
    text2speech(texto, idioma)
