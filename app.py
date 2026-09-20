"""Thin Flask launcher for the IoT Operations Agent."""
from iot_agent.web import create_app

app = create_app()


if __name__ == "__main__":
    print("IoT Operations Agent running at http://localhost:5001")
    app.run(host="0.0.0.0", port=5001, debug=False)
