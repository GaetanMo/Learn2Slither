
class Snake:
	def __init__ (self, start_pos):
		self.position = start_pos
		print(self.position)
	
	def move(self, direction):
		y, x = self.position[0]
		if direction == "UP":
			new_pos = (y - 1, x)
		if direction == "LEFT":
			new_pos = (y, x -1)
		if direction == "DOWN":
			new_pos = (y + 1, x)
		if direction == "RIGHT":
			new_pos = (y, x + 1)
		self.position = [new_pos] + self.position[:-1]
	
	def get_position(self):
		return self.position