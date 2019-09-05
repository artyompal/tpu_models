''' Some debug helpers. '''

import inspect, os, re
from typing import Any


def dline() -> int:
    frame = inspect.currentframe().f_back           # type: ignore
    return frame.f_lineno

def dprint(*args: Any) -> None:
    frame = inspect.currentframe().f_back           # type: ignore
    name = os.path.basename(inspect.getframeinfo(frame).filename)

    with open(inspect.getframeinfo(frame).filename) as f:
        line = f.readlines()[frame.f_lineno - 1]    # type: ignore

    m = re.match(r'\s*dprint\((.*)\)\s*', line)
    if m:
        print(f'{name}:{frame.f_lineno}', m.group(1), *args)
    else:
        print(f'{name}:{frame.f_lineno} dprint parse error', *args)

def assert_eq(*args: Any) -> None:
    assert len(args) == 2
    if args[0] != args[1]:
        frame = inspect.currentframe().f_back           # type: ignore
        name = os.path.basename(inspect.getframeinfo(frame).filename)

        with open(inspect.getframeinfo(frame).filename) as f:
            line = f.readlines()[frame.f_lineno - 1]    # type: ignore

        m = re.match(r'\s*(assert_eq\(.*\))\s*', line)
        if m:
            print(f'{name}:{frame.f_lineno} assertion failed:', m.group(1))
            print(f'{args[0]} != {args[1]}')

        assert False

def assert_ne(*args: Any) -> None:
    assert len(args) == 2
    if args[0] == args[1]:
        frame = inspect.currentframe().f_back           # type: ignore
        name = os.path.basename(inspect.getframeinfo(frame).filename)

        with open(inspect.getframeinfo(frame).filename) as f:
            line = f.readlines()[frame.f_lineno - 1]    # type: ignore

        m = re.match(r'\s*(assert_ne\(.*\))\s*', line)
        if m:
            print(f'{name}:{frame.f_lineno} assertion failed:', m.group(1))
            print(f'{args[0]} == {args[1]}')

        assert False
