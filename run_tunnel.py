#!/usr/bin/env python3
"""
Lance l'API FastAPI avec un tunnel ngrok public.

Sécurité (obligatoire) :
    - NGROK_AUTHTOKEN doit être défini en variable d'environnement.
      Récupération : https://dashboard.ngrok.com/get-started/your-authtoken
    - API_BEARER_TOKEN doit être défini en variable d'environnement (sinon
      l'API en génère un et l'affiche dans les logs au démarrage).

Usage:
    export NGROK_AUTHTOKEN=xxxxxxxxxxxxxxxxxxxx
    export API_BEARER_TOKEN=mon_token_secret
    python run_tunnel.py
    python run_tunnel.py --port 8001
"""
import argparse
import os
import signal
import sys
import threading
import time

import requests
import uvicorn
from pyngrok import ngrok, conf as ngrok_conf


def main():
    parser = argparse.ArgumentParser(description="API RAG + tunnel ngrok")
    parser.add_argument("--port", type=int, default=8001, help="Port local (défaut: 8001)")
    args = parser.parse_args()

    # 0. Vérifier que ngrok est configuré (authtoken)
    authtoken = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not authtoken:
        print(
            "ERREUR : la variable d'environnement NGROK_AUTHTOKEN n'est pas définie.\n"
            "Sans token, ngrok refuse les tunnels longs et n'importe qui peut "
            "scanner l'URL.\n"
            "  1. Crée un compte gratuit : https://dashboard.ngrok.com/signup\n"
            "  2. Récupère ton token   : https://dashboard.ngrok.com/get-started/your-authtoken\n"
            "  3. Exporte-le           : export NGROK_AUTHTOKEN=xxxxx (ou $env:NGROK_AUTHTOKEN sous PowerShell)",
            file=sys.stderr,
        )
        sys.exit(1)
    ngrok_conf.get_default().auth_token = authtoken

    # 1. Lancer FastAPI en thread background
    def run_server():
        uvicorn.run("api:app", host="0.0.0.0", port=args.port)

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # 2. Attendre que le serveur soit prêt — backoff exponentiel borné
    health_url = f"http://localhost:{args.port}/health"
    deadline = time.time() + 240  # 4 min max
    delay = 0.5
    while time.time() < deadline:
        try:
            r = requests.get(health_url, timeout=2)
            if r.status_code == 200:
                print("FastAPI ready!")
                break
        except requests.RequestException:
            pass
        time.sleep(delay)
        delay = min(delay * 1.5, 5.0)
    else:
        raise RuntimeError(f"Server failed to start on port {args.port} after 120s")

    # 3. Lancer ngrok
    public_url = ngrok.connect(args.port)
    print(f"Public URL: {public_url}")
    print("API + Tunnel OK")
    print("Rappel : toutes les requêtes doivent inclure 'Authorization: Bearer <API_BEARER_TOKEN>'")

    # Shutdown propre sur SIGINT / SIGTERM
    def shutdown(signum=None, frame=None):
        print("\nShutdown...")
        try:
            ngrok.disconnect(public_url)
            ngrok.kill()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    # Garde le process en vie
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
