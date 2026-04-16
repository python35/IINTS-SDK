from __future__ import annotations

import argparse
import logging
import os
import threading
from pathlib import Path

import uvicorn

from .api import create_patient_app
from .runtime import LivePatientDaemon, PatientRuntimeConfig


def _configure_logging(log_path: str) -> None:
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def _resolve_api_token(config: PatientRuntimeConfig) -> str | None:
    if config.api_token_env:
        token = os.getenv(config.api_token_env, "").strip()
        if not token:
            raise RuntimeError(
                f"API token environment variable '{config.api_token_env}' is not set or is empty."
            )
        return token
    if config.api_token_file:
        token_path = Path(config.api_token_file).expanduser().resolve()
        if not token_path.is_file():
            raise RuntimeError(f"API token file does not exist: {token_path}")
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise RuntimeError(f"API token file is empty: {token_path}")
        return token
    return None


def _start_api_server(config: PatientRuntimeConfig) -> tuple[uvicorn.Server, threading.Thread]:
    app = create_patient_app(config.workspace_path, api_token=_resolve_api_token(config))
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.api_host,
            port=config.api_port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True, name="iints-patient-api")
    thread.start()
    return server, thread


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the IINTS digital patient daemon.")
    parser.add_argument("--config", required=True, help="Path to patient runtime JSON config.")
    parser.add_argument("--reset", action="store_true", help="Reset the runtime instead of resuming from snapshot.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional maximum step count for testing.")
    args = parser.parse_args()

    config = PatientRuntimeConfig.from_path(Path(args.config))
    config.workspace_path.mkdir(parents=True, exist_ok=True)
    _configure_logging(str(config.log_path))

    daemon = LivePatientDaemon(config)
    daemon.install_signal_handlers()
    daemon.bootstrap(reset=bool(args.reset))
    server, thread = _start_api_server(config)
    daemon._server = server
    daemon._server_thread = thread
    try:
        daemon.run(max_steps=args.max_steps)
    finally:
        server.should_exit = True
        thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
