"""
Stub web server for unit tests
Copyright 2020 Microsoft
"""

from aiohttp import web
from .test_simulator_protocol import (
    _MOCK_EPISODE_FINISH_RESPONSE,
    _MOCK_EPISODE_START_RESPONSE,
    _MOCK_EPISODE_STEP_RESPONSE,
    _MOCK_UNREGISTER_RESPONSE,
    _MOCK_REGISTRATION_RESPONSE,
)


def count_me(fnc):
    """ decorator to count function calls in a class variable
    """

    def increment(self, *args, **kwargs):
        self._count += 1
        return fnc(self, *args, **kwargs)

    return increment


def get_app():
    stub = ServiceStub()
    app = web.Application()
    app.router.add_get("/", stub.root)
    app.router.add_post("/v2/workspaces/{workspace}/simulatorSessions", stub.register)
    app.router.add_post(
        "/v2/workspaces/{workspace}/simulatorSessions/{session_id}/advance",
        stub.get_next_event,
    )
    return app


def start_app() -> None:
    the_app = get_app()
    web.run_app(app=the_app, host="127.0.0.1", port=9000)


class ServiceStub:
    _FLAKY = False
    _count = 0
    _fail_point = 10
    _fail_duration = 8

    def __init__(self):
        pass

    def reset_flags(self):
        self._FLAKY = False
        self._count = 0

    async def root(self, request):
        return web.json_response({})

    async def register(self, request):
        self.reset_flags()
        if "unauthorized" in request.match_info["workspace"]:
            return web.Response(status=401, text="Unauthorized")
        if "forbidden" in request.match_info["workspace"]:
            return web.Response(status=403, text="Forbidden")
        if "badgateway" in request.match_info["workspace"]:
            return web.Response(status=502, text="BadGateway")
        if "unavailable" in request.match_info["workspace"]:
            return web.Response(status=503, text="Service Unavailable")
        if "gatewaytimeout" in request.match_info["workspace"]:
            return web.Response(status=504, text="GatewayTimeout")
        if "flaky" in request.match_info["workspace"]:
            self._FLAKY = True
        return web.json_response(_MOCK_REGISTRATION_RESPONSE)

    @count_me
    async def get_next_event(self, request):
        if (
            self._FLAKY
            and self._count > self._fail_point
            and self._count < self._fail_point + self._fail_duration
        ):
            return web.Response(status=503, text="Service Unavailable")

        if self._count == 1:
            return web.json_response(_MOCK_EPISODE_START_RESPONSE)

        if self._count % 25 == 0:
            return web.json_response(_MOCK_EPISODE_FINISH_RESPONSE)

        if "500" in request.match_info["workspace"]:
            return web.json_response(status=500, text="Internal Server Error")

        return web.json_response(_MOCK_EPISODE_STEP_RESPONSE)

    async def unregister(self, request):
        return web.json_response(_MOCK_UNREGISTER_RESPONSE)


if __name__ == "__main__":
    start_app()
