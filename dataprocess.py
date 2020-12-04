"""
Event and index format:
0 - 127 = note on value = (index)
128 - 256 = note off value = (index - 129) 
257 - 357 = time shift value = (index - 157) as seconds
358 - 362 = velocity value = (index - 357) * 25 as percent
Notes are in range (0, 127) inclusive
Time shift is in seconds, in increments of 10ms (10ms, 1000ms) inclusive
Velocity is in range (0, 127) inclusive
"""

from enum import Enum

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