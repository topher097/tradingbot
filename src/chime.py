import simpleaudio as sa
import numpy as np
import time


#---------------------
# Create and play noise to notify script has completed
# source = https://realpython.com/playing-and-recording-sound-python/#simpleaudio
#---------------------
def play_tone(freq=440, bursts=5, burst_time=0.1):
    frequency = freq        # Frequency of the tone
    fs = 44100              # Samples per second
    seconds = burst_time    # Note duration

    # Generate array with seconds*sample_rate steps, ranging between 0 and seconds
    t = np.linspace(0, seconds, int(seconds*fs), False)
    # Generate a x Hz sine wave
    note = np.sin(frequency * t * 2 * np.pi)
    # Ensure that highest value is in 16-bit range
    audio = note * (2**15 - 1) / np.max(np.abs(note))
    # Convert to 16-bit data
    audio = audio.astype(np.int16)

    for burst in range(0, bursts):
        # Start playback
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        # Wait for playback to finish before next loop
        play_obj.wait_done()
        time.sleep(seconds)

if __name__ == '__main__':
    play_tone()