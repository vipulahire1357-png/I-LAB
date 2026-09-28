class AppState:
    def __init__(self):
        self.occupancy = 0
        self.lights = False
        self.left_fans = False
        self.right_fans = False
        self.mode = "AUTO"


app_state = AppState()