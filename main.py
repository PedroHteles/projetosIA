import numpy as np
import os
os.environ['WRAPT_DISABLE_EXTENSIONS'] = 'True'
import wave
import matplotlib.pyplot as plt

from tensorflow.keras import models

from recording_helper import record_audio,encode_audio,decode_audio,save_audio
from tf_helper import preprocess_audiobuffer
import tensorflow as tf



# !! Modify this in the correct order
commands = ['cinco', 'dois' ,'quatro', 'seis' ,'sete', 'tres', 'um' ,'zero']
export_dir = 'saved_model/pedrao'

loaded_model = tf.saved_model.load(export_dir)
infer = loaded_model.signatures["serving_default"]


import tensorflow as tf

def get_spectrogram(waveform, sample_rate=16000, frame_length=1024, frame_step=256, fft_length=1024):
    input_len = sample_rate
    waveform = waveform[:input_len]

    # Preenchimento com zeros
    zero_padding = tf.zeros(
        [sample_rate] - tf.shape(waveform),
        dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)

    # Transformada de Fourier de Curto Tempo (STFT)
    spectrogram = tf.signal.stft(equal_length, frame_length=frame_length, frame_step=frame_step, fft_length=fft_length)

    # Cálculo da magnitude e normalização (entre 0 e 1)
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram / tf.math.reduce_max(spectrogram)  # Normalização

    # Adicionar dimensão de canal
    spectrogram = spectrogram[..., tf.newaxis]
    # Adicionar dimensão para o número de amostras
    spectrogram = spectrogram[tf.newaxis, ...]
    
    print("Shape do espectrograma:", spectrogram.shape)  # Imprime a forma do espectrograma
    return spectrogram





def plot_spectrogram(spectrogram, ax):
    # Remover dimensões adicionais, se existirem
    if len(spectrogram.shape) == 4:  # [1, time, freq, 1]
        spectrogram = np.squeeze(spectrogram, axis=0)
        spectrogram = np.squeeze(spectrogram, axis=-1)

    # Certificar-se de que o espectrograma tem duas dimensões [time, freq]
    if len(spectrogram.shape) == 3:
        spectrogram = np.squeeze(spectrogram, axis=-1)

    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)




def predict_mic():
# Capturar áudio em tempo real
    audio_data = record_audio()
    audio_binary = encode_audio(audio_data)
    decoded_audio = decode_audio(audio_binary)
    save_audio(audio_binary, filename='output.wav')
    spectrogram = get_spectrogram(decoded_audio)

    prediction = infer(spectrogram)
    # Acessar o tensor de saída usando a chave 'output_0'
    output_tensor = prediction['output_0']

    # Converter o tensor para um array NumPy
    output_array = output_tensor.numpy()
    predicted_index = np.argmax(output_array)
    print(commands[predicted_index]) 
    # fig, axes = plt.subplots(2, figsize=(12, 8))
    # timescale = np.arange(decoded_audio.shape[0])
    # axes[0].plot(timescale, decoded_audio.numpy())
    # axes[0].set_title('Waveform')
    # axes[0].set_xlim([0, 16000])

    # plot_spectrogram(spectrogram.numpy(), axes[1])
    # axes[1].set_title('Spectrogram')
    # plt.show()

        
if __name__ == "__main__":
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        # move_turtle(command)