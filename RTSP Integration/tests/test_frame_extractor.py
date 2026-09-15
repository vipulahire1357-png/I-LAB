import pytest
import time
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from frame_extractor import FrameExtractor
from data_models import FrameData

class MockRTSPClient:
    def __init__(self):
        self.is_connected = False
        self.connect_calls = 0
        self.read_calls = 0

    def connect(self):
        self.is_connected = True
        self.connect_calls += 1
        return True

    def read(self):
        self.read_calls += 1
        # Return a dummy frame (10x10 black image)
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        self.is_connected = False

    def reconnect_with_backoff(self):
        return self.connect()

def test_frame_extractor_fps():
    client = MockRTSPClient()
    # Extract at 2 FPS
    extract_config = {"fps": 2}
    output_config = {"save_to_disk": False}

    extractor = FrameExtractor("test_cam", client, extract_config, output_config, queue_size=5)
    extractor.start()

    # Let it run for 1 second
    time.sleep(1.0)
    extractor.stop()

    # The worker runs fast, so client.read() should be called many times
    assert client.read_calls > 10

    # At 2 FPS, in 1 second, it should yield approximately 2 frames.
    frames_retrieved = 0
    while True:
        frame = extractor.get_frame(timeout=0.1)
        if frame is None:
            break
        frames_retrieved += 1

    # Should be around 2 frames, allow 1 or 3 due to timing
    assert 1 <= frames_retrieved <= 3
