from flask import Flask, jsonify

app = Flask(__name__)


@app.route("/status")
def status():
    return jsonify({
        "occupancy": 18,
        "lights": True,
        "leftFans": True,
        "rightFans": False,
        "mode": "AUTO"
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)