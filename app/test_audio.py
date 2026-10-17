from reachy_mini import ReachyMini
import time

with ReachyMini(host='192.168.8.105', connection_mode='network', media_backend='default') as mini:
    mini.media.start_recording()
    time.sleep(3)
    sample = mini.media.get_audio_sample()
    print('Got audio sample:', sample is not None, type(sample))
