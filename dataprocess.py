import numpy as np
import pretty_midi
from enum import Enum
from operator import itemgetter
import math

time_step = 0.01

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
	358 - 377 = SET_VELOCITY
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
	# Vocab of 378 elements
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
	elif index in range(358, 378):
		# Return Set Velocity event
		return Event(EventType.SET_VELOCITY, (127 * (index - 357)) / 20)

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
		return int( ( ( (event.value * 20) / 127)+0.1) + 357)

def midi_array_to_event(midi_as_array):
	"""
	Take converted MIDI array and convert to array of Event objects
	"""
	# Sort MIDI array
	midi = np.array(sorted(midi_as_array, key=itemgetter(2)))
	# Init result
	result = []
	# Accumulators for computing start and end times
	active_notes = []
	curr_time = 0
	# For comparing velocities
	prev_vel_range = 0
	# For all the entries in the midi array
	for i in midi:
		# Add the current note
		active_notes.append(i)
		# Check the difference between the current time and the start of this note
		time_diff = i[2] - curr_time
		# If the difference is greater than 1, we need many time shifts
		if time_diff > 1:
			n = int(time_diff) # Number of full shifts
			r = int((time_diff - n) * 100) / 100 # Remainder shift value
			for t in range(n):
				result.append(Event(EventType.TIME_SHIFT, 1))
			result.append(Event(EventType.TIME_SHIFT, r))
			shift_value = n + r
		# If the difference is less than the greatest possible time step, shift by the time step
		# If its too low, consider it a simultaneous note
		elif time_step > time_diff >= 0.007:
			shift_value = time_step
			result.append(Event(EventType.TIME_SHIFT, shift_value))
		# Otherwise shift normally
		else:
			shift_value = int((i[2] - curr_time) * 100) / 100
			if shift_value > 0:
				result.append(Event(EventType.TIME_SHIFT, shift_value))
		# Accumulate shifted time
		curr_time += shift_value

		# Check if there are notes that are playing that need to end
		notes_to_end = [x for x in active_notes if curr_time >= x[3]]
		active_notes[:] = (x for x in active_notes if curr_time < x[3])
		# For the finished notes
		for j in notes_to_end:
			# End the note
			result.append(Event(EventType.NOTE_OFF, j[1]))

		# If the velocity has changed by a large enough amount, add a set velocity event
		temp_velocity = i[0]
		bin_size = (127/20)
		for vel in range(20):
			if temp_velocity < (vel + 1) * bin_size:
				if prev_vel_range != vel:
					result.append(Event(EventType.SET_VELOCITY, int((vel + 1) * bin_size)))
					prev_vel_range = vel
				break

		# Start the note
		result.append(Event(EventType.NOTE_ON, i[1]))

	# If there are still notes in midi_acc
	if active_notes:
		for i in active_notes:
			if i[3] > curr_time:
				# Check the difference between the current time and the end of this note
				time_diff = i[3] - curr_time
				# If the difference is greater than 1, we need many time shifts
				if time_diff > 1:
					n = int(time_diff) # Number of full shifts
					r = int((time_diff - n) * 100) / 100 # Remainder shift value
					for t in range(n):
						result.append(Event(EventType.TIME_SHIFT, 1))
					result.append(Event(EventType.TIME_SHIFT, r))
					shift_value = n + r
				# If the difference is less than the greatest possible time step, shift by the time step
				# If its too low, consider it a simultaneous note
				elif time_step > time_diff >= 0.007:
					shift_value = time_step
					result.append(Event(EventType.TIME_SHIFT, shift_value))
				# Otherwise shift normally
				else:
					shift_value = int((i[3] - curr_time) * 100) / 100
					if shift_value > 0:
						result.append(Event(EventType.TIME_SHIFT, shift_value))
				# Accumulate shifted time
				curr_time += shift_value
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
	# Debugging
	total_errors = 0

	for index, event in enumerate(events):
		if event.event_type is EventType.NOTE_ON:
			# If the note is present in the dictionary it will be added to the midi_arr
			if notes_on.get(event.value) is not None:
				midi_arr.append(pretty_midi.Note(velocity=int(curr_velocity), pitch=event.value, start=notes_on.get(event.value), end=curr_time))
			# Regardless we add/update the note into the dictionary
			notes_on.update({event.value : curr_time})
		elif event.event_type is EventType.NOTE_OFF:
			#Ensures the note has been turned off previously and sends a warning otherwise
			if notes_on.get(event.value) is not None:
				print("VEL:", curr_velocity)
				midi_arr.append(pretty_midi.Note(velocity=int(curr_velocity), pitch=event.value, start=notes_on.get(event.value), end=curr_time))
				notes_on.pop(event.value)
			else:
				print("Error: Note", str(event.value), "is trying to be turned off when it has never been turned on [", index, "]")
				total_errors += 1
		elif event.event_type is EventType.TIME_SHIFT:
			#Increments curr_time
			curr_time += event.value
		elif event.event_type is EventType.SET_VELOCITY:
			curr_velocity = event.value
		else:
			continue
	# If any of the notes in the dictionary haven't been turned off yet, we end them at the curr_time 
	for note in notes_on.keys():
		midi_arr.append(pretty_midi.Note(velocity=int(curr_velocity), pitch=note, start=notes_on.get(note), end=curr_time))
	
	print("TOTAL ERRORS:", total_errors)
	return midi_arr

def event_to_midi_array2(events):
	"""
	Take array of Event objects and convert to midi array
	"""
	# Holds the output midi array
	result = []
	# Holds the current velocity
	curr_velocity = 100
	# Holds the current time
	curr_time = 0
	# notes_on, contains notes that are currently on, {note:start_time}
	notes_on = {}
	# Debugging
	total_errors = 0

	for index, event in enumerate(events):
		# If this event changes the velocity
		if event.event_type is EventType.SET_VELOCITY:
			# Set the velocity
			curr_velocity = event.value
		# IF this is a time shifting event
		elif event.event_type is EventType.TIME_SHIFT:
			# Shift the time by the given value
			curr_time += event.value
		# If this event starts a note
		elif event.event_type is EventType.NOTE_ON:
			# Accumulate this note and await its end time
			notes_on[event.value] = [curr_velocity, curr_time]
		# If this event ends a note
		elif event.event_type is EventType.NOTE_OFF:
			# Verify that this note can be ended
			if notes_on.get(event.value) is not None:
				# Add it to the result
				vel = notes_on[event.value][0]
				start_time = notes_on[event.value][1]
				result.append(pretty_midi.Note(velocity=int(vel), pitch=int(event.value), start=start_time, end=curr_time))
				# Remove it from the dictionary
				del notes_on[event.value]
			# If it cannot be ended, show a warning
			else:
				print("Error: Note", str(event.value), "is trying to be turned off when it has never been turned on [", index, "]")
				total_errors += 1
	print("TOTAL ERRORS:", total_errors)
	# Return the completed array
	return result