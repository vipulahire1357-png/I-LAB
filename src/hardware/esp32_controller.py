import requests

from src.hardware.esp32_discovery import discover_esp32


class ESP32Controller:

    def __init__(self, ip_address=None):
        self.ip_address = ip_address

        # Last state successfully sent to ESP32
        self.last_state = None

    def set_state(self,lights,fan1,fan2,fan3, fan4):

        # Create current state
        current_state = (
            int(lights),
            int(fan1),
            int(fan2),
            int(fan3),
            int(fan4)

                )

        # If nothing changed, don't send another request
        if current_state == self.last_state:
            return

        # If we don't have an IP, try discovery
        if self.ip_address is None:
            print("ESP32 IP not available. Searching...")
            self.ip_address = discover_esp32()

            if self.ip_address is None:
                print("ESP32 not found.")
                return

        # ESP32 control URL
        url = f"http://{self.ip_address}/control"

        # Data to send
        params = {
            "lights": current_state[0],
            "fan1": current_state[1],
            "fan2": current_state[2],
            "fan3": current_state[3],
            "fan4": current_state[4],
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
                f"FAN1: {current_state[1]} | "
                f"FAN2: {current_state[2]} | "
                f"FAN3: {current_state[3]} | "
                f"FAN4: {current_state[4]} | "
                f"Status: {response.status_code}"
            )

            # Save state only if request succeeded
            if response.status_code == 200:
                self.last_state = current_state

        except requests.RequestException as e:

            print(f"ESP32 connection failed: {e}")

            # Try to find ESP32 again
            print("Searching for ESP32 again...")

            new_ip = discover_esp32()

            if new_ip is not None:

                self.ip_address = new_ip

                print(f"ESP32 new IP found: {self.ip_address}")

                # Retry the same command once
                try:
                    response = requests.get(
                        f"http://{self.ip_address}/control",
                        params=params,
                        timeout=2
                    )

                    print(
                        f"ESP32 retry → Status: {response.status_code}"
                    )

                    if response.status_code == 200:
                        self.last_state = current_state

                except requests.RequestException as retry_error:
                    print(
                        f"ESP32 retry failed: {retry_error}"
                    )

            else:
                print("ESP32 could not be rediscovered.")