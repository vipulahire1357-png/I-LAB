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
        self.fan1_state = False
        self.fan2_state = False
        self.fan3_state = False
        self.fan4_state = False

        # Empty timers
        self.lights_empty_since = None
        self.fan1_empty_since = None
        self.fan2_empty_since = None
        self.fan3_empty_since = None
        self.fan4_empty_since = None

    def update(self,total_occupancy,left_occupied,right_occupied,fan1_occupied,fan2_occupied,fan3_occupied,fan4_occupied):
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
        # FAN 1
        # -----------------------------

        if fan1_occupied:
            self.fan1_state = True
            self.fan1_empty_since = None
        else:
            if self.fan1_empty_since is None:
                self.fan1_empty_since = current_time

            if current_time - self.fan1_empty_since >= self.EMPTY_TIMEOUT:
                self.fan1_state = False

        # -----------------------------
        # FAN 2
        # -----------------------------

        if fan2_occupied:
            self.fan2_state = True
            self.fan2_empty_since = None
        else:
            if self.fan2_empty_since is None:
                self.fan2_empty_since = current_time

            if current_time - self.fan2_empty_since >= self.EMPTY_TIMEOUT:
                self.fan2_state = False
        # -----------------------------
        # FAN 3
        # -----------------------------

        if fan3_occupied:
            self.fan3_state = True
            self.fan3_empty_since = None
        else:
            if self.fan3_empty_since is None:
                self.fan3_empty_since = current_time

            if current_time - self.fan3_empty_since >= self.EMPTY_TIMEOUT:
                self.fan3_state = False
        # -----------------------------
        # FAN 4
        # -----------------------------

        if fan4_occupied:
            self.fan4_state = True
            self.fan4_empty_since = None
        else:
            if self.fan4_empty_since is None:
                self.fan4_empty_since = current_time

            if current_time - self.fan4_empty_since >= self.EMPTY_TIMEOUT:
                self.fan4_state = False
        # -----------------------------
        # SEND STATE TO ESP32
        # -----------------------------

        self.esp32.set_state(
        lights=self.lights_state,
        fan1=self.fan1_state,
        fan2=self.fan2_state,
        fan3=self.fan3_state,
        fan4=self.fan4_state
    )
        # app_state.lights = self.lights_state
        # app_state.left_fans = self.left_fans_state
        # app_state.right_fans = self.right_fans_state