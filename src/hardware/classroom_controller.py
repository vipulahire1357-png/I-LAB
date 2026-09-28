import time

from src.hardware.esp32_controller import ESP32Controller
from src.app_state import app_state


class ClassroomController:

    # Testing ke liye 10 seconds
    # Final mein: 5 * 60
    EMPTY_TIMEOUT = 10

    def __init__(self):
        self.esp32 = ESP32Controller()

        # Current hardware states
        self.lights_state = False
        self.left_fans_state = False
        self.right_fans_state = False

        # Empty timers
        self.lights_empty_since = None
        self.left_empty_since = None
        self.right_empty_since = None

    def update(self, total_occupancy, left_occupied, right_occupied):
        if app_state.mode != "AUTO":
            return

        current_time = time.time()

        # -----------------------------
        # LIGHTS
        # -----------------------------

        if total_occupancy > 0:

            self.lights_state = True
            self.lights_empty_since = None

        else:

            # Start empty timer
            if self.lights_empty_since is None:
                self.lights_empty_since = current_time

            # Turn OFF only after timeout
            if current_time - self.lights_empty_since >= self.EMPTY_TIMEOUT:
                self.lights_state = False

        # -----------------------------
        # LEFT FANS
        # -----------------------------

        if left_occupied:

            self.left_fans_state = True
            self.left_empty_since = None

        else:

            # Start empty timer
            if self.left_empty_since is None:
                self.left_empty_since = current_time

            # Turn OFF only after timeout
            if current_time - self.left_empty_since >= self.EMPTY_TIMEOUT:
                self.left_fans_state = False

        # -----------------------------
        # RIGHT FANS
        # -----------------------------

        if right_occupied:

            self.right_fans_state = True
            self.right_empty_since = None

        else:

            # Start empty timer
            if self.right_empty_since is None:
                self.right_empty_since = current_time

            # Turn OFF only after timeout
            if current_time - self.right_empty_since >= self.EMPTY_TIMEOUT:
                self.right_fans_state = False

        # -----------------------------
        # SEND STATE TO ESP32
        # -----------------------------

        self.esp32.set_state(
            lights=self.lights_state,
            left_fans=self.left_fans_state,
            right_fans=self.right_fans_state
        )
        app_state.lights = self.lights_state
        app_state.left_fans = self.left_fans_state
        app_state.right_fans = self.right_fans_state