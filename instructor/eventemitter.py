# eventemitter.py

from enum import Enum
from collections import defaultdict
from typing import Any, Callable, List, Type, TypeVar, overload, Literal

T = TypeVar("T")


class EventName(Enum):
    """Enumeration of event names."""

    COMPLETION_AFTER = "completion:after"
    COMPLETION_ERROR = "completion:error"
    # Add other event names as needed


class EventEmitter:
    """A class that manages event handlers and emits events."""

    def __init__(self):
        # A dictionary mapping event names to a list of handler functions
        self._handlers: defaultdict[EventName, List[Callable[[Any], None]]] = (
            defaultdict(list)
        )

    @overload
    def on(
        self,
        event_name: Literal[EventName.COMPLETION_AFTER] | Literal["completion:after"],
        handler: Callable[[Any], None],
    ) -> None: ...

    @overload
    def on(
        self,
        event_name: Literal[EventName.COMPLETION_ERROR] | Literal["completion:error"],
        handler: Callable[[Any], None],
    ) -> None: ...

    def on(self, event_name: EventName, handler: Callable[[Any], None]) -> None:
        """
        Registers an event handler for a specific event.

        :param event_name: The name of the event to listen for.
        :param handler: A callable that handles the event.
        """
        self._handlers[event_name].append(handler)

    def emit(self, event_name: EventName, event_data: Any) -> None:
        """
        Emits an event, invoking all registered handlers with the event data.

        :param event_name: The name of the event to emit.
        :param event_data: The data associated with the event.
        """
        for handler in self._handlers.get(event_name, []):
            handler(event_data)

    def off(self, event_name: EventName, handler: Callable[[Any], None]) -> None:
        """
        Removes a specific handler from an event.

        :param event_name: The name of the event.
        :param handler: The handler to remove.
        """
        if event_name in self._handlers:
            self._handlers[event_name].remove(handler)
            if not self._handlers[event_name]:
                del self._handlers[event_name]

    def clear(self, event_name: EventName = None) -> None:
        """
        Clears handlers for a specific event or all events.

        :param event_name: The name of the event to clear handlers for. If None, all handlers are cleared.
        """
        if event_name is not None:
            self._handlers.pop(event_name, None)
        else:
            self._handlers.clear()
