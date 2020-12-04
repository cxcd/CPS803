from enum import Enum
from operator import itemgetter
import pretty_midi
import numpy as np

class EventType(Enum):
    """
    Enum to hold all possible events
    """
    NOTE_ON = 1,
    NOTE_OFF = 2,
    TIME_SHIFT = 3,
    SET_VELOCITY = 4

class Event():
    """
    Class to associate an event type to a value
    event_type = The type of event this represents
    value = the value with respect to the type
    i.e. NOTE_ON, NOTE_OFF have pitch values (0, 127) inclusive
         TIME_SHIFT has a time value in seconds (0.01, 1) inclusive
         SET_VELOCITY has a velocity (0, 127) inclusive
    The Index Format is as follows:
    0 - 127 = NOTE_ON
    128 - 256 = NOTE_OFF
    257 - 357 = TIME_SHIFT
    358 - 362 = SET_VELOCITY
    """
    def __init__(self, _type, _value):
        self.event_type = _type
        self.value = _value
    def __repr__(self):
        return "(" + self.event_type.name + " : " + str(self.value) + ")\n"

def index_to_event(index):
    """
    Convert an index to its respective Event object in the format.
    """
    # Vocab of 362 elements
    # Range is not inclusive, goes from a to b - 1
    if index in range(0, 128):
        # Return Note On event
        return Event(EventType.NOTE_ON, index)
    elif index in range(128, 257):
        # Return Note Off event
        return Event(EventType.NOTE_OFF, index - 128)
    elif index in range(257, 358):
        # Return Time Shift event
        return Event(EventType.TIME_SHIFT, (index - 157) / 100)
    elif index in range(358, 362):
        # Return Set Velocity event
        return Event(EventType.SET_VELOCITY, (127 * (index - 357)) / 4)

def event_to_index(event):
    """
    Convert an Event object to its respective index in the format.
    Reverses all operations from index_to_event
    """
    if event.event_type is EventType.NOTE_ON:
        return event.value
    elif event.event_type is EventType.NOTE_OFF:
        return event.value + 128
    elif event.event_type is EventType.TIME_SHIFT:
        return int((event.value * 100) + 157)
    elif event.event_type is EventType.SET_VELOCITY:
        return int(((event.value * 4) / 127) + 357)

def midi_array_to_event(midi_as_array):
    """
    Take converted MIDI array and convert to array of Event objects
    """
    # Get MIDI data
    #midi = pretty_midi.PrettyMIDI(midi_path).instruments[0].notes
    # Sort by start times TODO is this reduntant?
    midi = sorted(midi_as_array, key=itemgetter(2))
    # Init result
    result = []
    # Accumulators for computing start and end times
    midi_acc = []
    curr_time = 0
    # For comparing velocities
    prev_vel_range = 0
    # For all the entries in the midi array
    for i in midi:
        # Add the current note
        midi_acc.append(i)
        # If the start time is greater than the current time
        if i[2] > curr_time:
            # Shift time, truncate to hundreths place
            shift_value = int((i[2] - curr_time) * 100) / 100
            result.append(Event(EventType.TIME_SHIFT, shift_value))
            # Accumulate shifted time
            curr_time += shift_value
            # Check if there are notes that are playing that need to end
            notes_to_end = (x for x in midi_acc if curr_time >= x[3])
            midi_acc[:] = (x for x in midi_acc if curr_time < x[3])
            # For the finished notes
            for j in notes_to_end:
                # End the note
                result.append(Event(EventType.NOTE_OFF, j[1]))
        # If the velocity has changed by a large enough amount, add a set velocity event
        if (0 <= i[0] < 31.75 and not prev_vel_range is 1):
            result.append(Event(EventType.SET_VELOCITY, i[0]))
            prev_vel_range = 1
        elif (31.75 <= i[0] < 42.33 and not prev_vel_range is 2):
            result.append(Event(EventType.SET_VELOCITY, i[0]))
            prev_vel_range = 2
        elif (42.33 <= i[0] < 63.5 and not prev_vel_range is 3):
            result.append(Event(EventType.SET_VELOCITY, i[0]))
            prev_vel_range = 3
        elif (63.5 <= i[0] < 128 and not prev_vel_range is 4):
            result.append(Event(EventType.SET_VELOCITY, i[0]))
            prev_vel_range = 4
        # Start the note
        result.append(Event(EventType.NOTE_ON, i[1]))
    # If there are still notes in midi_acc
    if midi_acc:
        for i in midi_acc:
            # Shift time to meet the ends and end them
            shift_value = int((i[3] - curr_time) * 100) / 100
            result.append(Event(EventType.TIME_SHIFT, shift_value))
            result.append(Event(EventType.NOTE_OFF, i[1]))
    # Return array
    return result