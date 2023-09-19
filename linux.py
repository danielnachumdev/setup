"""danielnachumdev's linux quick setup script
"""
import functools
import subprocess
import os
import sys
from enum import Enum
from threading import Semaphore, Lock, Thread
from abc import ABC, abstractmethod
from typing import IO, Generator, Tuple, Union, Iterable, Any
from pathlib import Path
DEBUG: bool = True


class OSType(Enum):
    """enum result for possible results of get_os()
    """
    LINUX = "Linux"
    WINDOWS = "Windows"
    OSX = "OS X"
    UNKNOWN = "Unknown"


def get_os() -> OSType:
    """returns the type of operation system running this code

    Returns:
        OSType: enum result
    """
    p = sys.platform
    if p in {"linux", "linux2"}:
        return OSType.LINUX
    if p == "darwin":
        return OSType.OSX
    if p == "win32":
        return OSType.WINDOWS
    return OSType.UNKNOWN


def generator_from_stream(stream: Union[IO, Iterable[Any]]) -> Generator[Any, None, None]:
    """will yield values from a given stream

    Args:
        stream (IO): the stream

    Yields:
        Generator[Any, None, None]: the resulting generator
    """
    for v in stream:
        yield v


def atomic(func):
    """will make function thread safe by making it
    accessible for only one thread at one time

    Args:
        func (Callable): function to make thread safe

    Returns:
        Callable: the thread safe function
    """
    lock = Lock()

    @ functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        with lock:
            return func(*args, **kwargs)
    return wrapper


def threadify(func):
    """will modify the function that when calling it a new thread
    will start to run it with provided arguments.\nnote that no return value will be given

    Args:
        func (Callable): the function to make a thread

    Returns:
        Callable: the modified function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        Thread(target=func, args=args, kwargs=kwargs).start()
    return wrapper


def join_generators(*generators) -> Generator[Tuple[int, Any], None, None]:
    """will join generators to yield from all of them simultaneously 
    without busy waiting, using semaphores and multithreading 

    Yields:
        Generator[Any, None, None]: one generator that combines all of the given ones
    """
    queue = Queue()
    edit_queue_semaphore = Semaphore(1)
    queue_status_semaphore = Semaphore(0)
    finished_threads_counter = AtomicCounter()

    @threadify
    def thread_entry_point(index: int, generator: Generator) -> None:
        for value in generator:
            with edit_queue_semaphore:
                queue.push((index, value))
            queue_status_semaphore.release()
        finished_threads_counter.increment()

        if finished_threads_counter.get() == len(generators):
            # re-release the lock once from the last thread because it
            # gets stuck in the main loop after the generation has stopped
            queue_status_semaphore.release()

    for i, generator in enumerate(generators):
        thread_entry_point(i, generator)

    while finished_threads_counter.get() < len(generators):
        queue_status_semaphore.acquire()
        with edit_queue_semaphore:
            # needed for the very last iteration of the "while" loop. see above comment
            if not queue.is_empty():
                yield queue.pop()
    with edit_queue_semaphore:
        for value in queue:
            yield value


def cmrt(*args, shell: bool = True) -> Generator[Tuple[int, bytes], None, None]:
    """Executes a command and yields stdout and stderr in real-time.

    Args:
        shell (bool, optional): If True, the command is executed through the shell. Defaults to True.

    Raises:
        TypeError: if 'shell' is not boolean

    Yields:
        Generator[tuple[int, bytes], None, None]: the tuple yielded will contain the 'stream identifier'
            0 - stdout,
            1 - stderr
        and the actual value from the stream
    """
    if not isinstance(shell, bool):
        raise TypeError("The 'shell' parameter must be of type bool.")

    # Quote the arguments that represent file or directory paths.
    for i, arg in enumerate(args):
        path_obj = Path(args[i])
        if path_obj.is_file() or path_obj.is_dir():
            args = (*args[:i], f"\"{arg}\"", *args[i+1:])

    # Join the arguments into a command string and execute the command.
    cmd = " ".join(args)

    with subprocess.Popen(cmd, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        combined = join_generators(
            generator_from_stream(process.stdout),  # type:ignore
            generator_from_stream(process.stderr)  # type:ignore
        )
        for tup in combined:
            yield tup


class IndentedWriter:
    LOG_PREFIX = "[setup]: "

    def __init__(self, streams: list[IO[str]] = [sys.stdout], indent: str = '--> ') -> None:
        self._streams = streams
        self._indent = indent
        self._amount: int = 0

    def indent(self) -> None:
        self._amount += 1

    def undent(self) -> None:
        self._amount = max(self._amount-1, 0)

    def _write(self, msg) -> None:
        for stream in self._streams:
            stream.write(msg)

    def log(self, *args, sep: str = ' ', end: str = '\n') -> None:
        msg = self._indent * self._amount + \
            IndentedWriter.LOG_PREFIX + sep.join(args)+end
        self._write(msg)

    def print(self, *args, sep: str = ' ', end: str = '\n'):
        msg = self._indent * self._amount+sep.join(args)+end
        self._write(msg)


LOGGER = IndentedWriter()


class Queue:
    """classic Queue data structure"""

    def __init__(self) -> None:
        self.data: list = []

    def pop(self) -> Any:
        """return the oldest element while removing it from the queue

        Returns:
            Any: result
        """
        return self.data.pop()

    def push(self, value: Any) -> None:
        """adds a new element to the queue

        Args:
            value (Any): the value to add
        """
        self.data.insert(0, value)

    def peek(self) -> Any:
        """returns the oldest element in the queue 
        without removing it from the queue

        Returns:
            Any: result
        """
        return self.data[-1]

    def __len__(self) -> int:
        return len(self.data)

    def is_empty(self) -> bool:
        """returns whether the queue is empty

        Returns:
            bool: result
        """
        return len(self) == 0

    def __str__(self) -> str:
        return repr(self)

    def __repr__(self) -> str:
        return str(self.data)

    def __iter__(self):
        return iter(self.data)

    def push_many(self, arr: list):
        """will push many objects to the Queue

        Args:
            arr (list): the objects to push
        """
        for v in arr:
            self.push(v)


class AtomicCounter:
    """An atomic counter class
    """

    def __init__(self, initial_value: Union[int, float] = 0, increment_amount: Union[int, float] = 1) -> None:
        self.value = initial_value
        self.increment_value = increment_amount

    @atomic
    def increment(self) -> None:
        """increments the stored value by the increment amount
        """
        self.value += self.increment_value

    @atomic
    def decrement(self) -> None:
        """decrements the stored value by the increment amount
        """
        self.value -= self.increment_value

    @atomic
    def get(self) -> Union[int, float]:
        """returns the current value of the counter

        Returns:
            Union[int, float]: value
        """
        return self.value

    @atomic
    def set(self, value: Union[int, float]):
        """sets the values of the counter

        Args:
            value (Union[int, float]): value to set
        """
        self.value = value


class Exmplainable(ABC):
    @abstractmethod
    def explain(self) -> None: ...


class Executable(Exmplainable):
    @abstractmethod
    def execute(self) -> None: ...

    def explain(self) -> None:
        LOGGER.log(f"(Executing) {self}")


class Installable(Executable):
    """An object which can be installed
    """

    def execute(self) -> None:
        self.install()

    @abstractmethod
    def install(self) -> None:
        """installs the Installable
        """

    def explain(self) -> None:
        LOGGER.log(f"(Installing) {self}")


class TerminalCommand(Executable):
    def __init__(self, cmd: str) -> None:
        self._cmd = cmd

    def execute(self) -> None:
        if DEBUG:
            return
        LOGGER.indent()
        for tup in cmrt(self._cmd):
            stream_num, msg = tup
            LOGGER.print(msg.decode(), end='')
        LOGGER.undent()

    def explain(self) -> None:
        LOGGER.log(f"(Executing) {self._cmd}")


class AptTarget(Installable):
    def __init__(self, name: str, auto_accept: bool = True, fix_missing: bool = True) -> None:
        self._name = name
        cmd = f'sudo apt-get install {name}'
        if auto_accept:
            cmd += ' -y'

        if fix_missing:
            cmd += ' --fix-missing'
        self._executor = TerminalCommand(cmd)

    def install(self) -> None:
        self._executor.execute()

    def explain(self) -> None:
        LOGGER.log(f'(Downloading) {self._name}')


class FileAppender(Executable):
    def __init__(self, filepath: str, lines: list[str], only_if_not_exists: bool = True) -> None:
        self._og_filepath = filepath
        self._filepath = os.path.expanduser(filepath)
        self._lines = lines
        self._only_if_not_exists = only_if_not_exists

    def execute(self) -> None:
        LOGGER.indent()
        if DEBUG:
            for line in self._lines:
                LOGGER.log(f"(Appended) {line}")
            LOGGER.undent()
            return
        lines_set: set[str] = set()
        with open(self._filepath, 'r', encoding='utf8') as f:
            for line in f:
                lines_set.add(line.strip())
        with open(self._filepath, 'a', encoding='utf8') as f:
            for line in self._lines:
                if self._only_if_not_exists:
                    if line in lines_set:
                        LOGGER.log(f"(Already Exists) {line}")
                        continue
                LOGGER.log(f"Appended) {line}")
                f.write(line+"\n")
        LOGGER.undent()

    def explain(self) -> None:
        LOGGER.log(f"(Modifying) {self._og_filepath}")


class InstallTarget(Installable):
    def __init__(self, display_name: str, cmds: list[Executable]) -> None:
        self._display_name = display_name
        self._cmds = cmds

    def install(self) -> None:
        LOGGER.indent()
        for cmd in self._cmds:
            cmd.explain()
            cmd.execute()
        LOGGER.undent()

    def explain(self) -> None:
        LOGGER.log(f"(Installing) {self._display_name}")


class WriteMessage(Executable):
    def __init__(self, writing_func, msg: str) -> None:
        self._msg = msg
        self._writing_func = writing_func

    def execute(self) -> None:
        self._writing_func(self._msg)

    def explain(self) -> None:
        pass


def main():
    dct = globals()

    LOGGER.log('Starting the installer')
    if get_os() != OSType.LINUX:
        LOGGER.log("This script should only be run on linux. exiting...")
        exit(1)

    LOGGER.log(
        "Elevating access rights, if necessary please type in your password")
    is_elevated: bool = False
    for tup in cmrt('sudo echo'):
        if tup != (0, b'\n'):
            break
    else:
        is_elevated = True

    if not is_elevated:
        LOGGER.print("Failed to elevate privilages, try again")
        exit(1)

    executables: list[Executable] = [
        AptTarget('build-essential'),
        AptTarget('git'),
        AptTarget('gcc'),
        AptTarget('g++'),
        AptTarget('cmake'),
        AptTarget('make'),
        InstallTarget("redis", [
            TerminalCommand(
                'curl -fsSL https://packages.redis.io/gpg | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg'),
            TerminalCommand(
                'echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list'),
            TerminalCommand('sudo apt-get update'),
            AptTarget('redis'),
            AptTarget('redis-server')
        ]),
        InstallTarget("Modular", [
            TerminalCommand(
                'curl https://get.modular.com | MODULAR_AUTH=mut_55c07900b0634760a280ce48dcdb3262 sh -'),
            TerminalCommand('modular install mojo'),
            FileAppender(
                '~/.bashrc', ['export MODULAR_HOME="$HOME/.modular"', 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"']),
            TerminalCommand('source ~/.bashrc')
        ]),
        AptTarget('rabbitmq-server'),
        InstallTarget("Anaconda3", [
            WriteMessage(
                LOGGER.log, "(NOTICE) Installing Anaconda3 takes a long time"),
            TerminalCommand(
                'wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh'),
            TerminalCommand('bash Anaconda3-2023.07-2-Linux-x86_64.sh')
        ]),
        TerminalCommand('sudo apt-get update'),
        TerminalCommand('sudo apt-get upgrade')
    ]

    for executable in executables:
        executable.explain()
        executable.execute()
    LOGGER.print()
    LOGGER.print()
    LOGGER.log("Finished installing!")


if __name__ == '__main__':
    main()
