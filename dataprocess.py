import pretty_midi
from enum import Enum
from operator import itemgetter
import math

time_step = 0.01
bin_size = (127/20)

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
		return "(" + self.event_type.name + " : " + str(self.value) + ")"

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


def get_shift_value(time_diff):
	shift_values = [] # The shift values if there is more than one
	shift_sum = 0 # The sum of all shift values (the shift value if there is only one)
	# If the difference is greater than 1, we need many time shifts
	if time_diff > 1:
		n = int(time_diff) # Number of full shifts
		r = int((time_diff - n) * 100) / 100 # Remainder shift value
		for t in range(n):
			shift_values.append(1)
		shift_values.append(r)
		shift_sum = n + r
	# If the difference is less than the greatest possible time step, shift by the time step
	# If its too low, consider it a simultaneous note
	elif time_step > time_diff >= 0.007:
		shift_sum = time_step
	# Otherwise shift normally
	else:
		shift_sum = int((time_diff) * 100) / 100
	return shift_values, shift_sum

def midi_array_to_event(midi_as_array):
	"""
	Take converted MIDI array and convert to array of Event objects
	"""
	# Sort MIDI array
	midi = sorted(midi_as_array, key=itemgetter(2))
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

		# Get time shift values 
		shift_values, shift_sum = get_shift_value(i[2] - curr_time)
		# Apply time shift to the next start note
		if shift_values:
			for s in shift_values:
				if s > 0:
					result.append(Event(EventType.TIME_SHIFT, s))
		else:
			result.append(Event(EventType.TIME_SHIFT, shift_sum))
		# Update time
		curr_time += shift_sum

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
				# Apply time shift
				shift_values, shift_sum = get_shift_value(i[3] - curr_time)
				if shift_values:
					for s in shift_values:
						if s > 0:
							result.append(Event(EventType.TIME_SHIFT, s))
				else:
					result.append(Event(EventType.TIME_SHIFT, shift_sum))
				# Update time
				curr_time += shift_sum
			# End note
			result.append(Event(EventType.NOTE_OFF, i[1]))
	
	# Return array
	return result

# THIS ONE LOOKS LIKE IT WORKS BUT MAYBE IT DOESNT ???
def midi_array_to_event2(midi_as_array):
	"""
	Take converted MIDI array and convert to array of Event objects
	"""
	# Sort MIDI array
	midi = sorted(midi_as_array, key=itemgetter(2))
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

		# Get time shift values up to the start of this note
		shift_values, shift_sum = get_shift_value(i[2] - curr_time)
		future_time = curr_time + shift_sum
		
		# Check if there are notes that are playing that end in between this time and the current time
		notes_to_end = [x for x in active_notes if future_time >= x[3]]
		active_notes[:] = (x for x in active_notes if future_time < x[3])
		# For the notes that will finish
		for j in notes_to_end:
			# Shift up to the end of that note
			end_shift_values, end_shift_sum = get_shift_value(j[3] - curr_time)
			if end_shift_values:
				for s in end_shift_values:
					if s > 0:
						result.append(Event(EventType.TIME_SHIFT, s))
			else:
				if shift_sum > 0:
					result.append(Event(EventType.TIME_SHIFT, end_shift_sum))
			# Update time
			curr_time += end_shift_sum
			# End the note
			result.append(Event(EventType.NOTE_OFF, j[1]))

		# Shift the time up to the start of the current note
		shift_values, shift_sum = get_shift_value(i[2] - curr_time)
		if shift_values:
			for s in shift_values:
				if s > 0:
					result.append(Event(EventType.TIME_SHIFT, s))
		else:
			if shift_sum > 0:
				result.append(Event(EventType.TIME_SHIFT, shift_sum))
		# Update time
		curr_time += shift_sum

		# If the velocity has changed by a large enough amount, add a set velocity event
		temp_velocity = i[0]
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
				# Apply time shift
				shift_values, shift_sum = get_shift_value(i[3] - curr_time)
				if shift_values:
					for s in shift_values:
						if s > 0:
							result.append(Event(EventType.TIME_SHIFT, s))
				else:
					if shift_sum > 0:
						result.append(Event(EventType.TIME_SHIFT, shift_sum))
				# Update time
				curr_time += shift_sum
			# End note
			result.append(Event(EventType.NOTE_OFF, i[1]))
	
	# Return array
	return result


def event_to_midi_array(events):
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
		print("E:", event)
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
		else:
			print("Error: Object is not an Event")
			total_errors += 1
	print("TOTAL ERRORS:", total_errors)
	# Return the completed array
	return result