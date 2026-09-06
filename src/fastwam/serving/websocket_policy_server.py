"""WebSocket request/reply transport for policy inference."""

from __future__ import annotations

import asyncio
import http
import logging
import traceback
from typing import Any

import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from fastwam.serving import msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serve one policy over binary MessagePack messages.

    Metadata is sent when a client connects. Each subsequent binary message
    contains one observation and receives one inference result. A single active
    client is enforced to give it exclusive use of the inference model.
    """

    def __init__(
        self,
        policy: Any,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = str(host)
        self._port = int(port)
        self._metadata = metadata or {}
        self._client_active = False
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self._run())

    async def _run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            logger.info("Policy server listening on ws://%s:%d", self._host, self._port)
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        if self._client_active:
            message = "Another policy client is already active."
            await websocket.send(message)
            await websocket.close(code=1013, reason=message)
            return

        self._client_active = True
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        try:
            self._policy.reset()
            await websocket.send(packer.pack(self._metadata))
            while True:
                raw = await websocket.recv()
                if isinstance(raw, str):
                    raise TypeError("Inference requests must be binary MessagePack messages.")
                observation = msgpack_numpy.unpackb(raw)
                if not isinstance(observation, dict):
                    raise TypeError(
                        f"Decoded observation must be a dict, got {type(observation).__name__}."
                    )
                result = self._policy.infer(observation)
                await websocket.send(packer.pack(result))
        except websockets.ConnectionClosed:
            logger.info("Connection from %s closed", websocket.remote_address)
        except Exception:
            error = traceback.format_exc()
            logger.error("Policy inference failed:\n%s", error)
            try:
                await websocket.send(error)
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous message.",
                )
            except websockets.ConnectionClosed:
                pass
            raise
        finally:
            try:
                self._policy.reset()
            except Exception:
                logger.exception("Failed to reset policy after client disconnect")
            self._client_active = False


def _health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
