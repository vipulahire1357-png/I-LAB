import requests


class ESP32Controller:

    def __init__(self, ip_address):
        self.ip_address = ip_address

        # Last state successfully sent to ESP32
        self.last_state = None

    def set_state(self, lights, left_fans, right_fans):

        # Create current state
        current_state = (
            int(lights),
            int(left_fans),
            int(right_fans)
        )

        # If nothing changed, don't send another request
        if current_state == self.last_state:
            return

        # ESP32 control URL
        url = f"http://{self.ip_address}/control"

        # Data to send
        params = {
            "lights": current_state[0],
            "left": current_state[1],
            "right": current_state[2],
        }

        try:
            # Send request to ESP32
            response = requests.get(
                url,
                params=params,
                timeout=2
            )

            print(
                f"ESP32 → Lights: {current_state[0]} | "
                f"Left Fans: {current_state[1]} | "
                f"Right Fans: {current_state[2]} | "
                f"Status: {response.status_code}"
            )

            # Save state only if request succeeded
            self.last_state = current_state

        except requests.RequestException as e:

            # ESP32 unavailable → don't stop the model
            print(f"ESP32 connection failed: {e}")


# Temporary test
if __name__ == "__main__":

    esp32 = ESP32Controller("192.168.43.75")

    esp32.set_state(
        lights=1,
        left_fans=0,
        right_fans=0
    )