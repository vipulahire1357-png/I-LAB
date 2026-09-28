import socket

DISCOVERY_PORT = 4210
DISCOVERY_MESSAGE = "ILAB_DISCOVER"


def discover_esp32(timeout=3):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Allow UDP broadcast
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

    # Wait maximum 3 seconds
    sock.settimeout(timeout)

    try:
        print("Searching for I-LAB ESP32...")

        sock.sendto(
            DISCOVERY_MESSAGE.encode(),
            ("255.255.255.255", DISCOVERY_PORT)
        )

        while True:
            data, address = sock.recvfrom(1024)

            message = data.decode()

            if message == "ILAB_ESP32":
                esp32_ip = address[0]

                print(f"ESP32 found: {esp32_ip}")

                return esp32_ip

    except socket.timeout:
        print("No I-LAB ESP32 found.")
        return None

    finally:
        sock.close()

# if __name__ == "__main__":
#     discover_esp32()