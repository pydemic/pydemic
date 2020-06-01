import datetime
from functools import total_ordering
from numbers import Number
from typing import Iterator, Dict, TYPE_CHECKING

import sidekick as sk

from .base_attr import BaseAttr
from ..utils import fmt, as_seq

NOT_GIVEN = object()
if TYPE_CHECKING:
    pass


@total_ordering
class Event(sk.Record):
    """
    Represents an Event.
    """

    key: str
    time: int
    date: datetime.date
    description: str = ""
    tags: set = frozenset()

    @classmethod
    def from_model(cls, model, key, time_or_date, description=None, tags=()):
        """
        Create an event instance passing model and only a single day or date.
        """
        if isinstance(time_or_date, (float, int)):
            time = time_or_date
            date = model.to_date(time)
        else:
            date = time_or_date
            time = model.to_time(date)
        description = description or key.replace("_", " ").title()
        return Event(key, time, date, description=description, tags=set(tags))

    def __str__(self):
        return f"{self.description} ({fmt(self.date)})"

    def __eq__(self, other):
        if isinstance(other, Event):
            return self[:-1] == other[:-1]
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Event):
            return self.time > other.time
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Event):
            return self.time >= other.time
        return NotImplemented

    def tag(self, tags):
        """
        Tag event with the given tag or list of tags.

        Notes:
            The event tags set is a publicly exposed attribute and can be used
            to make queries such as the examples:

             * ``tag in ev.tags``: check if event has tag.
             * ``ev.tags.issuperset(seq)``: check if event has all tags in sequence.
             * ``ev.tags & set(seq)``: check tags has some tags in seq.
        """
        self.tags.update(as_seq(tags))


class Info(BaseAttr):
    """
    Info objects store static information about a simulation.

    Not all information exposed by this object is necessarily relevant to a
    simulation model, but might be useful for later analysis or for
    convenience.

    Info attributes are organized in a dotted namespace.
    """

    __slots__ = ()
    _method_namespace = "info"

    @property
    def _cache(self):
        # noinspection PyProtectedMember
        return self.model._info_cache

    def clear(self):
        self._cache.clear()

    def save_event(self, key, time_or_date=None, description=None):
        """
        Save event under the given key.

        Args:
            key:
                A unique identifier for the event.
            time_or_date:
                When the event happened. If not given, assumes the current
                instant.
            description:
                An optional description for the event.
        """
        if time_or_date is None:
            time_or_date = self.model.time
        event = Event.from_model(self.model, key, time_or_date, description)
        self._set_item("event", key, event)

    def get_events(self, *args, tags=False) -> Iterator[Event]:
        """
        Return all events that happened in the given day or date or interval.

        * info.get_events(day) -> events in a specific day
        * info.get_events(a, b) -> events between a and b (inclusive)
        * info.get_events(a, ...) -> events beginning at a
        * info.get_events(..., b) -> events up to b
        """
        events = self["events"]
        model = self.model

        if len(args) == 1:
            seq = select_events_at(events, model.to_time(args[0]))
        elif args == 2:
            args = map(model.to_time, args)
            seq = select_events_between(events, *args)
        else:
            raise TypeError("method requires 1 or 2 arguments")

        if tags:
            tags = set(tags)
            for ev in seq:
                if tags.issubset(ev.tags):
                    yield ev
        else:
            yield from seq

    def _set_item(self, group, key, value):
        if group == "event" and not isinstance(value, Event):
            if isinstance(value, (int, datetime.date)):
                value = Event.from_model(self.model, key, value)
            else:
                value = Event.from_model(self.model, key, *value)

        super()._set_item(group, key, value)


def select_events_at(evs: Dict[str, Event], a: Number) -> Iterator[Event]:
    return (ev for k, ev in evs if ev.time == a)


def select_events_between(evs: Dict[str, Event], a, b) -> Iterator[Event]:
    if a is ... and b is ...:
        return iter(evs.values())
    elif b is ...:
        return (ev for k, ev in evs if a <= ev.time)
    elif a is ...:
        return (ev for k, ev in evs if ev.time <= b)
    else:
        return (ev for k, ev in evs if a <= ev.time <= b)
