import sounddevice as sd
import soundfile as sf
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
import librosa

# Загружаем аудиофайл и читаем его содержимое в сигнал и частоту дискретизации
audio_file = 'test-audio.mp3'
signal, sample_rate = sf.read(audio_file, dtype='float32')

# Воспроизводим аудиосигнал и ожидаем завершения воспроизведения
sd.play(signal, sample_rate)
status = sd.wait()

# Рассчитываем продолжительность аудиофайла в секундах
duration = len(signal) / sample_rate
# Определяем количество каналов (моно или стерео)
num_channels = signal.shape[1] if len(signal.shape) > 1 else 1
# Устанавливаем битовую глубину (предполагается 16 бит)
bit_depth = 16

# Выводим информацию о продолжительности, частоте дискретизации, количестве каналов и битовой глубине
print(f"Duration: {duration:.2f} seconds")
print(f"Sample Rate: {sample_rate} Hz")
print(f"Channels: {num_channels}")
print(f"Bit Depth: {bit_depth} bits")

# Нормализуем сигнал по максимальной амплитуде
signal = signal / np.max(np.abs(signal), axis=0)

# Создаем временную ось для сигнала и строим график аудиосигнала
time = np.linspace(0, duration, len(signal))
plt.plot(time, signal)
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('Audio Signal')
plt.show()

# Вычисляем диапазон амплитуд сигнала и амплитуды шума в начале и конце сигнала
amplitude_range = np.max(signal) - np.min(signal)
noise_amplitude_start = np.mean(signal[:1000])
noise_amplitude_end = np.mean(signal[-1000:])

# Выводим значения диапазона амплитуд и амплитуды шума
print(f"Amplitude Range: {amplitude_range}")
print(f"Noise Amplitude (start): {noise_amplitude_start}")
print(f"Noise Amplitude (end): {noise_amplitude_end}")

# Функция для вычисления энергии сигнала
def energy(signal):
    return np.sum(signal ** 2)

# Функция для подсчета количества пересечений нуля
def zero_crossings(signal):
    return np.sum(np.abs(np.diff(np.sign(signal)))) / 2

# Устанавливаем длительность окна в 10 мс и вычисляем количество отсчетов в окне
window_duration = 0.01
window_samples = int(window_duration * sample_rate)
num_windows = len(signal) // window_samples

# Инициализируем списки для хранения энергий и пересечений нуля
energies = []
zero_crossings_list = []

# Разбиваем сигнал на окна и вычисляем энергию и пересечения нуля для каждого окна
for i in range(num_windows):
    start = i * window_samples
    end = start + window_samples
    window = signal[start:end]

    if len(window) == window_samples:
        energies.append(energy(window))
        zero_crossings_list.append(zero_crossings(window))

# Нормализуем энергии и пересечения нуля
energies = np.array(energies) / np.max(energies)
zero_crossings_list = np.array(zero_crossings_list) / np.max(zero_crossings_list)

# Строим графики энергий и пересечений нуля
plt.figure(figsize=(10, 6))
plt.plot(energies, label='Energy (E)', color='red')
plt.plot(zero_crossings_list, label='Zero Crossings (Z)', color='blue')
plt.xlabel('Window Index')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()

# Берем сегмент сигнала длиной 2048 отсчетов
segment = signal[:2048]

# Применяем окно Хэмминга к сегменту сигнала
hamming_window = np.hamming(len(segment))
windowed_segment = segment * hamming_window

# Выполняем БПФ и вычисляем амплитудный спектр
fft_result = scipy.fftpack.fft(windowed_segment)
amplitude_spectrum = np.log(np.abs(fft_result))

# Строим график амплитудного спектра
frequencies = np.linspace(0, sample_rate, len(amplitude_spectrum))
plt.plot(frequencies[:10000], amplitude_spectrum[:10000])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Log Amplitude')
plt.title('Amplitude Spectrum')
plt.show()

# Вычисляем коэффициенты LPC
lpc_coeffs = librosa.lpc(segment, order=20)

# Создаем расширенный массив коэффициентов LPC
extended_lpc_coeffs = np.zeros_like(segment)
extended_lpc_coeffs[:len(lpc_coeffs)] = lpc_coeffs

# Вычисляем спектр LPC
lpc_spectrum = np.log(np.abs(np.fft.fft(extended_lpc_coeffs)))
lpc_spectrum = -1 * lpc_spectrum

# Строим графики оригинального спектра и спектра LPC
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:10000], amplitude_spectrum[:10000], label='Original Spectrum')
plt.plot(frequencies[:10000], lpc_spectrum[:10000], label='LPC Spectrum')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Log Amplitude')
plt.legend()
plt.show()
