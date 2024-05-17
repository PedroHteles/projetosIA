import sounddevice as sd
import numpy as np
import tensorflow as tf
import noisereduce as nr

def record_audio(duration=1, sample_rate=16000):
    print("Gravando áudio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Gravação concluída.")
    return audio_data

def encode_audio(audio_data, sample_rate=16000):
    audio_data = np.squeeze(audio_data)  # Remover a dimensão extra para compatibilidade
    # Supondo que a parte inicial do áudio (primeiros 0.5 segundos) seja apenas ruído
    noise_sample = audio_data[:int(0.5 * sample_rate)]
    reduced_noise_audio = nr.reduce_noise(y=audio_data, y_noise=noise_sample, sr=sample_rate)
    
    audio_data = tf.convert_to_tensor(reduced_noise_audio, dtype=tf.float32)
    audio_data = tf.expand_dims(audio_data, axis=-1)  # Expandir a dimensão para (amostras, 1)
    audio_binary = tf.audio.encode_wav(audio_data, sample_rate=sample_rate)
    return audio_binary

def decode_audio(audio_binary):
    audio_tensor = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio_tensor.audio, axis=-1)

def save_audio(audio_binary, filename='output.wav'):
    tf.io.write_file(filename, audio_binary)
    print(f"Áudio salvo como {filename}")

