import numpy as np
from enum import Enum
from operator import itemgetter

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
        return Event(EventType.TIME_SHIFT, (index - 257) / 100)
    elif index in range(358, 362):
        # Return Set Velocity event
        return Event(EventType.SET_VELOCITY, (127 * (index - 357)) / 4)

def event_to_index(event):
    """
    Convert an Event object to its respective index in the format.
    Reverses all operations from index_to_event
    """
    if event.event_type is EventType.NOTE_ON:
        return int(event.value)
    elif event.event_type is EventType.NOTE_OFF:
        return int(event.value + 128)
    elif event.event_type is EventType.TIME_SHIFT:
        return int((event.value * 100) + 257)
    elif event.event_type is EventType.SET_VELOCITY:
        return int(((event.value * 4) / 127) + 357)

def midi_array_to_event(midi_as_array):
    """
    Take converted MIDI array and convert to array of Event objects
    """
    # Sort MIDI array
    midi = np.array(sorted(midi_as_array, key=itemgetter(2)))
    # Offset the start times for music that starts late
    offset = midi[0][2] - 0.5
    if offset > 0:
        midi[:,2] -= offset
        midi[:,3] -= offset
    print("ARRAY:\n",midi)
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
        # If the start time is greater than or equal to the current time
        if i[2] > curr_time:
            # Shift time, truncate to hundreths place
            shift_value = int((i[2] - curr_time) * 100) / 100
            result.append(Event(EventType.TIME_SHIFT, shift_value))
            # Accumulate shifted time
            curr_time += shift_value
            # Check if there are notes that are playing that need to end
            notes_to_end = [x for x in midi_acc if curr_time >= x[3]]
            midi_acc[:] = (x for x in midi_acc if curr_time < x[3])
            # For the finished notes
            for j in notes_to_end:
                # End the note
                result.append(Event(EventType.NOTE_OFF, j[1]))
        # If the velocity has changed by a large enough amount, add a set velocity event
        if (0 <= i[0] < 31.75 and prev_vel_range != 1):
            result.append(Event(EventType.SET_VELOCITY, 31.75))
            prev_vel_range = 1
        elif (31.75 <= i[0] < 63.5 and prev_vel_range != 2):
            result.append(Event(EventType.SET_VELOCITY, 42.33))
            prev_vel_range = 2
        elif (63.5 <= i[0] < 95.25 and prev_vel_range != 3):
            result.append(Event(EventType.SET_VELOCITY, 63.5))
            prev_vel_range = 3
        elif (95.25 <= i[0] < 128 and prev_vel_range != 4):
            result.append(Event(EventType.SET_VELOCITY, 127))
            prev_vel_range = 4
        # Start the note
        result.append(Event(EventType.NOTE_ON, i[1]))
    # If there are still notes in midi_acc
    if midi_acc:
        for i in midi_acc:
            if i[3] > curr_time:
                # Shift time to meet the ends and end them
                shift_value = int((i[3] - curr_time) * 100) / 100
                curr_time += shift_value
                result.append(Event(EventType.TIME_SHIFT, shift_value))
            result.append(Event(EventType.NOTE_OFF, i[1]))
    # Return array
    return result


def event_to_midi_array(events):
    """
    Take array of Event objects and convert to midi array
    """
    # Holds the output midi array
    midi_arr = []
    # Holds the current velocity
    curr_velocity = 100
    # Holds the current time
    curr_time = 0
    # notes_on, contains notes that are currently on, {note:start_time}
    notes_on = {}

    for event in events:
        if event.event_type is EventType.NOTE_ON:
            # If the note is present in the dictionary it will be added to the midi_arr
            if notes_on.get(event.value) is not None:
                midi_arr.append(pretty_midi.Note(velocity=int(curr_velocity), pitch=event.value, start=notes_on.get(event.value), end=curr_time))
            # Regardless we add/update the note into the dictionary
            notes_on.update({event.value:curr_time})
        elif event.event_type is EventType.NOTE_OFF:
            #Ensures the note has been turned off previously and sends a warning otherwise
            if notes_on.get(event.value) is not None:
                midi_arr.append(pretty_midi.Note(velocity=int(curr_velocity), pitch=event.value, start=notes_on.get(event.value), end=curr_time))
                notes_on.pop(event.value)
            else:
                print("Error: Note "+str(event.value)+" is trying to be turned off when it has never been turned on")
        elif event.event_type is EventType.TIME_SHIFT:
            #Increments curr_time
            curr_time += event.value
        elif event.event_type is EventType.SET_VELOCITY:
            curr_velocity = event.value
    
    # If any of the notes in the dictionary haven't been turned off yet, we end them at the curr_time 
    for note in notes_on.keys():
        midi_arr.append(pretty_midi.Note(velocity=int(curr_velocity), pitch=note, start=notes_on.get(note), end=curr_time))
    
    return midi_arr
                
