from __future__ import annotations

import argparse
import asyncio
import json
import threading
import webbrowser

from aiohttp import WSMsgType, web

from .config import DecoderConfig, ROOT
from .runtime import DemoRuntime


class DemoServer:
    def __init__(self, runtime: DemoRuntime):
        self.runtime = runtime
        self.clients: set[web.WebSocketResponse] = set()
        self.runtime_task: asyncio.Task | None = None

    async def start(self, app: web.Application) -> None:
        self.runtime.initialize()
        self.runtime_task = asyncio.create_task(self.runtime.run(self.broadcast))

    async def stop(self, app: web.Application) -> None:
        self.runtime.running = False
        if self.runtime_task is not None:
            await self.runtime_task
        for client in list(self.clients):
            await client.close()
        self.runtime.close()

    async def broadcast(self, state: dict[str, object]) -> None:
        if not self.clients:
            return
        payload = json.dumps(state, separators=(",", ":"))
        stale = []
        for client in list(self.clients):
            try:
                await client.send_str(payload)
            except ConnectionError:
                stale.append(client)
        for client in stale:
            self.clients.discard(client)

    async def websocket(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        self.clients.add(socket)
        await socket.send_json(self.runtime.snapshot())
        try:
            async for message in socket:
                if message.type != WSMsgType.TEXT:
                    continue
                try:
                    await self.handle_command(json.loads(message.data))
                except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
                    await socket.send_json({"type": "error", "message": str(error)})
        finally:
            self.clients.discard(socket)
        return socket

    async def handle_command(self, data: dict[str, object]) -> None:
        if data.get("type") != "command":
            raise ValueError("Expected a command message")
        command = data["command"]
        if command == "pause":
            self.runtime.paused = bool(data.get("value", True))
            await self.broadcast(self.runtime.snapshot())
        elif command == "reset":
            self.runtime.reset()
            await self.broadcast(self.runtime.snapshot())
        elif command == "set_target":
            self.runtime.world.set_target(float(data["x"]), float(data["z"]))
        elif command == "toggle_orbit":
            self.runtime.world.target.orbit = bool(data.get("value", False))
        elif command == "spawn_obstacle":
            self.runtime.world.spawn_obstacle(float(data.get("side", 0.0)))
        elif command == "manual_stimulus":
            self.runtime.set_manual_stimulus(
                str(data["kind"]), float(data.get("duration", 1.0))
            )
        elif command == "set_speed":
            self.runtime.config.fixed_speed = max(0.0, min(float(data["value"]), 3.0))
        elif command == "single_step":
            if self.runtime.paused:
                state = None
                for _ in range(self.runtime.config.publish_steps):
                    state = self.runtime.step() or state
                if state is not None:
                    await self.broadcast(state)
        else:
            raise ValueError(f"Unknown command: {command}")

    async def latest_log(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(self.runtime.logger.path)


def build_app(device: str = "auto", seed: int = 1) -> web.Application:
    runtime = DemoRuntime(DecoderConfig.load(), device=device, seed=seed)
    server = DemoServer(runtime)
    app = web.Application()
    app["demo_server"] = server
    app.on_startup.append(server.start)
    app.on_cleanup.append(server.stop)
    app.router.add_get("/ws", server.websocket)
    app.router.add_get("/api/logs/latest", server.latest_log)

    three_build = ROOT / "node_modules" / "three" / "build"
    three_module = three_build / "three.module.js"
    if not three_module.exists():
        raise RuntimeError("Three.js is missing. Run `npm install` in the project directory.")
    app.router.add_static("/vendor/", three_build)
    app.router.add_get("/", lambda request: web.FileResponse(ROOT / "web" / "index.html"))
    app.router.add_static("/", ROOT / "web", show_index=True)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="FlyBrain closed-loop control demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(f"http://{args.host}:{args.port}",)).start()
    web.run_app(build_app(args.device, args.seed), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
