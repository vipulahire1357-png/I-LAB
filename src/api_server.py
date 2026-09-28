from flask import Flask, jsonify ,request
from src.app_state import app_state
from src.hardware.esp32_controller import ESP32Controller


app = Flask(__name__)
esp32 = ESP32Controller()

@app.route("/mode", methods=["POST"])
def set_mode():
    data = request.get_json()

    mode = data.get("mode")
    print("MODE REQUEST =", mode)

    if mode not in ["AUTO", "MANUAL"]:
        return jsonify({"error": "Invalid mode"}), 400

    app_state.mode = mode
    print("APP STATE MODE =", app_state.mode)

    return jsonify({
        "mode": app_state.mode
    })
@app.route("/status")
def status():
    return jsonify({
        "occupancy": app_state.occupancy,
        "lights": app_state.lights,
        "leftFans": app_state.left_fans,
        "rightFans": app_state.right_fans,
        "mode": app_state.mode
    })


@app.route("/control", methods=["POST"])
def manual_control():
    data = request.get_json()

    if app_state.mode != "MANUAL":
        return jsonify({"error": "System is in AUTO mode"}), 400

    lights = data.get("lights")
    left_fans = data.get("leftFans")
    right_fans = data.get("rightFans")

    if lights is not None:
        app_state.lights = bool(lights)

    if left_fans is not None:
        app_state.left_fans = bool(left_fans)

    if right_fans is not None:
        app_state.right_fans = bool(right_fans)

    esp32.set_state(
        lights=app_state.lights,
        left_fans=app_state.left_fans,
        right_fans=app_state.right_fans
    )

    return jsonify({
        "lights": app_state.lights,
        "leftFans": app_state.left_fans,
        "rightFans": app_state.right_fans
    })
def start_api_server():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    start_api_server()