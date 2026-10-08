from __future__ import annotations

import argparse
import asyncio

from aiohttp import ClientSession


async def receive_until(socket, predicate, messages: int = 30):
    for _ in range(messages):
        state = await socket.receive_json(timeout=5)
        if predicate(state):
            return state
    raise RuntimeError("Expected state was not observed within the message limit")


async def probe(url: str) -> None:
    async with ClientSession() as session:
        async with session.ws_connect(url) as socket:
            first = await socket.receive_json(timeout=5)
            second = await socket.receive_json(timeout=5)
            print(
                "connected",
                f"device={first['system']['device']}",
                f"control_hz={second['system']['control_hz']}",
                f"seq={second['seq']}",
                f"source={second['input']['source']}",
                f"turn={second['decoder']['filtered_turn']}",
            )
            await socket.send_json({
                "type": "command",
                "command": "manual_stimulus",
                "kind": "target_left",
                "duration": 1.0,
            })
            steering = await receive_until(
                socket,
                lambda state: state["input"]["source"] == "manual:target_left"
                and state["brain"]["rates_hz"]["steer_left"]
                > state["brain"]["rates_hz"]["steer_right"]
                and state["decoder"]["filtered_turn"] < -0.05,
            )
            print(
                "steering_ok",
                f"left_hz={steering['brain']['rates_hz']['steer_left']}",
                f"turn={steering['decoder']['filtered_turn']}",
            )

            await socket.send_json({
                "type": "command",
                "command": "manual_stimulus",
                "kind": "odor_left",
                "duration": 1.0,
            })
            odor = await receive_until(
                socket,
                lambda state: state["input"]["source"] == "manual:odor_left"
                and state["brain"]["rates_hz"]["odor_left"]
                > state["brain"]["rates_hz"]["odor_right"]
                and state["decoder"]["odor_turn"] < -0.05,
            )
            print(
                "odor_ok",
                f"left_hz={odor['brain']['rates_hz']['odor_left']}",
                f"odor_turn={odor['decoder']['odor_turn']}",
            )

            await socket.send_json({
                "type": "command",
                "command": "manual_stimulus",
                "kind": "loom_left",
                "duration": 1.0,
            })
            escape = await receive_until(
                socket,
                lambda state: state["input"]["source"] == "manual:loom_left"
                and state["decoder"]["action"],
            )
            print(
                "escape_ok",
                f"left_hz={escape['brain']['rates_hz']['escape_left']}",
                f"action={escape['decoder']['action']}",
            )
            await socket.send_json({"type": "command", "command": "reset"})
            reset_initial = await receive_until(socket, lambda state: state["seq"] == 0)
            reset_state = reset_initial
            for _ in range(12):
                reset_state = await socket.receive_json(timeout=5)
            assert reset_state is not None
            realtime_factor = reset_state["system"]["realtime_factor"]
            if not 0.8 <= realtime_factor <= 1.2:
                raise RuntimeError(f"Unexpected realtime factor after reset: {realtime_factor}")
            if reset_state["world"]["fly"]["y"] <= reset_initial["world"]["fly"]["y"]:
                raise RuntimeError("The fly did not gain altitude after reset")
            print(
                "reset_ok",
                f"realtime_factor={realtime_factor}",
                f"altitude={reset_state['world']['fly']['y']:.3f}",
            )

            await socket.send_json({
                "type": "command",
                "command": "spawn_obstacle",
                "side": 0,
            })
            looming = await receive_until(
                socket,
                lambda state: state["input"]["source"] == "world"
                and max(state["input"]["loom_left"], state["input"]["loom_right"]) > 0.1
                and state["decoder"]["action"],
                messages=50,
            )
            print(
                "world_loom_ok",
                f"loom={max(looming['input']['loom_left'], looming['input']['loom_right']):.3f}",
                f"action={looming['decoder']['action']}",
            )
            outcome = await receive_until(
                socket,
                lambda state: not state["world"]["obstacle"]["enabled"]
                and state["world"]["stats"]["threat_trials"] >= 1,
                messages=50,
            )
            if outcome["world"]["stats"]["threats_avoided"] < 1:
                raise RuntimeError("The closed-loop threat trial ended in contact")
            print(
                "avoidance_ok",
                f"rate={outcome['world']['stats']['avoidance_success_rate']:.3f}",
                f"altitude={outcome['world']['fly']['y']:.3f}",
            )
            await socket.send_json({"type": "command", "command": "reset"})
            await receive_until(socket, lambda state: state["seq"] == 0)
            print("final_reset_ok")


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe a running FlyBrain demo server")
    parser.add_argument("--url", default="http://127.0.0.1:8765/ws")
    args = parser.parse_args()
    asyncio.run(probe(args.url))


if __name__ == "__main__":
    main()
