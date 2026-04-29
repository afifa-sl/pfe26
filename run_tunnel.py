#!/usr/bin/env python3
"""
Lance l'API FastAPI avec un tunnel ngrok public.

Usage:
    python run_tunnel.py
    python run_tunnel.py --port 8001
"""
import threading
import time
import argparse

import requests
import uvicorn
from pyngrok import ngrok


def main():
    parser = argparse.ArgumentParser(description="API RAG + tunnel ngrok")
    parser.add_argument("--port", type=int, default=8001, help="Port local (défaut: 8001)")
    args = parser.parse_args()

    # 1. Lancer FastAPI en thread background
    def run_server():
        uvicorn.run("api:app", host="0.0.0.0", port=args.port)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # 2. Attendre que le serveur soit prêt
    health_url = f"http://localhost:{args.port}/health"
    for _ in range(60):
        try:
            r = requests.get(health_url)
            if r.status_code == 200:
                print("FastAPI ready!")
                break
        except requests.ConnectionError:
            pass
        time.sleep(1)
    else:
        raise RuntimeError("Server failed to start")

    # 3. Lancer ngrok
    public_url = ngrok.connect(args.port)
    print(f"Public URL: {public_url}")
    print("API + Tunnel OK")

    # Garde le process en vie
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nShutdown...")
        ngrok.disconnect(public_url)


if __name__ == "__main__":
    main()
