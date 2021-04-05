import asyncio
from asyncio.subprocess import PIPE
from asyncio import create_subprocess_exec
from typing import Any


async def _read_stream(stream: Any, callback: Any) -> Any:
    while True:
        line = await stream.readline()
        if line:
            callback(line)
        else:
            break


async def run(command: str) -> None:
    command_split = command.split(" ")
    process = await create_subprocess_exec(*command_split, stdout=PIPE, stderr=PIPE)

    await asyncio.wait(
        [
            _read_stream(
                process.stdout,
                lambda x: print("STDOUT: {}".format(x.decode("UTF8"))),
            ),
            _read_stream(
                process.stderr,
                lambda x: print("STDERR: {}".format(x.decode("UTF8"))),
            ),
        ],
    )

    await process.wait()
