from enum import Enum
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

def pretty_midi_to_event(midi_path):
    """
    Convert MIDI file to array of Event objects
    """
    # Get MIDI data
    midi = pretty_midi.PrettyMIDI(midi_path).instruments[0].notes
    # Init result
    result = []
    # Accumulators for computing start and end times
    midi_acc = []
    curr_time = 0
    # For all the entries in the midi array
    for i in midi:
        # Add the current note
        midi_acc.append(i)
        # If the start time is greater than the current time
        if i.start > curr_time:
            # TODO Shift time
            # TODO Accumulate shifted time
            # Check if there are notes that are playing that need to end
            notes_to_end = (x for x in midi_acc if curr_time >= x.end)
            midi_acc[:] = (x for x in midi_acc if curr_time < x.end)
            # For the finished notes
            for j in notes_to_end:
                # End the note
                result.append(Event(EventType.NOTE_OFF, j.pitch))
        # TODO if the velocity has changed, add a set velocity event
        # Start the note
        result.append(Event(EventType.NOTE_ON, i.pitch))
    # TODO check if there are still notes in midi_acc
    # TODO if there are, shift time to meet the ends and end them
    # Return array
    return result

"""
ALGORITHM FOR CONVERTING PRETTY_MIDI TO MT INDICES

sort notes by start time ascending

note_dict = { note : start, end }
current_time = 0

data = []

# This adds indices that correspond to events in a dictionary/list
# i.e. if it says 'start the note', add the index of the corresponding note's start event
# e.g. start the note E, add index 40 (or whatever it is) which correspondes to NOTE_START(E)

for each note in notes - 1:

    add current note to the dictionary

    if note.start_time > current_time:
        shift time by the difference between current_time and note.start_time, rounded to the desired prescision
        accumulate the shift in current_time
        
        if the current time is equal or greater than the end time of notes in the dictionary:
            end that note
            remove it from the dictionary
            
    if velocity changes by a big enough value, add a velocity change event

    start the note
    
    
for the last note:
    check dictionary for greatest end time
    shift time to meet the end
    then end all those notes

assert empty dictionary

save this array of events to a file
"""