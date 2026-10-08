import unittest
import asyncio
from types import SimpleNamespace

from app.server import DemoServer
from app.runtime import DemoRuntime


class BroadcastTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_during_send_does_not_abort_world_loop(self):
        server = DemoServer(None)

        class DisconnectingClient:
            async def send_str(self, payload):
                server.clients.discard(self)

        server.clients.update([DisconnectingClient(), DisconnectingClient()])
        await server.broadcast({"type": "state"})
        self.assertEqual(len(server.clients), 0)

    async def test_overdue_simulation_yields_to_network_tasks(self):
        runtime = SimpleNamespace(config=SimpleNamespace(dt=1e-9), world=SimpleNamespace(time=0.),
                                  generation=0, paused=False, running=False)
        network_ran = False

        async def network_task():
            nonlocal network_ran
            network_ran = True

        def step():
            runtime.world.time += .02
            if runtime.world.time >= .1:
                runtime.running = False
            return {}

        async def callback(state):
            pass

        runtime.step = step
        task = asyncio.create_task(network_task())
        await DemoRuntime.run(runtime, callback)
        try:
            self.assertTrue(network_ran)
        finally:
            await task
