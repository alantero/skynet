import pyaudio
import wave

import noisereduce as nr
import librosa
import soundfile as sf

import speech_recognition as sr
import whisper

# Define the basic parameters for the audio recording
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels (1 for mono)
RATE = 44100              # Sample rate (samples per second)
CHUNK = 1024              # Number of frames per buffer
RECORD_SECONDS = 5        # Duration of recording
WAVE_OUTPUT_FILENAME = "output.wav"  # Output filename


def record_voice():
    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open stream for recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording...")

    frames = []

    # Read data from stream
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    # Terminate the PortAudio interface
    p.terminate()

    # Save the recorded data as a WAV file
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



def noise_reduction():
    # Cargar el audio y la tasa de muestreo con librosa
    audio, rate = librosa.load('output.wav', sr=None)

    # Seleccionar una porción del audio donde solo hay ruido de fondo
    # Asumiendo que los primeros 0.5 segundos del archivo solo contienen ruido
    noise_clip = audio[0:int(0.5 * rate)]

    # Aplicar la reducción de ruido sobre el audio completo
    audio_clean = nr.reduce_noise(y=audio,sr=rate)

    # Guardar el audio limpio en un nuevo archivo WAV
    sf.write('output_clean.wav', audio_clean, rate)


def audio2text():

    # Inicializa el reconocedor
    r = sr.Recognizer()

    # Carga el archivo de audio
    audio_file = "output_clean.wav"

    with sr.AudioFile(audio_file) as source:
        # Escucha el archivo de audio
        audio_data = r.record(source)
        # Intenta reconocer el audio usando el reconocedor de Google
        try:
            text = r.recognize_google(audio_data, language="es-ES")  # Usa "en-US" para inglés
            print("Transcripción: " + text)
        except sr.UnknownValueError:
            print("Google Speech Recognition no pudo entender el audio")
        except sr.RequestError as e:
            print(f"No se pudo solicitar resultados desde el servicio de Google Speech Recognition; {e}")

    return voice2text


def audio2whisper():
    model = whisper.load_model("base")
    result = model.transcribe("output_clean.wav", fp16=False)
    #print("Whisper:", result["text"])
    return result["text"]


def voice2text():
    record_voice()
    noise_reduction()
    #voice2text()
    return audio2whisper()


if __name__ == "__main__":
    voice2text()
