"""
Start the demo GUI and open it in the browser. Press "play" on this file.

Needs: pip install fastapi uvicorn
Port: `gui: {port: 8080}` in configs/configs.yaml (default 8080).
"""
import os
import threading
import webbrowser

import uvicorn

import support

CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "configs.yaml")


def main():
    gui = (support.read_yaml(CONFIG_FILE) or {}).get("gui") or {}
    host, port = gui.get("host", "127.0.0.1"), int(gui.get("port", 8080))
    url = f"http://{'localhost' if host in ('0.0.0.0', '127.0.0.1') else host}:{port}"
    threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"Demo GUI on {url}  (Ctrl+C to stop)")
    uvicorn.run("web.app:app", host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
