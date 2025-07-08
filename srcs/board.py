import numpy as np
from snake import Snake
import tkinter as tk

class Board:
	def __init__(self):
		self.map = np.full((10, 10), 0, dtype=object)
		self.place_items()
		self.place_snake()
		self.init_window()

	def init_window(self):

		self.cell_size = 100
		self.root = tk.Tk()
		self.root.title("Learn2Slither 🐍")

		self.canvas = tk.Canvas(
			self.root,
			width=self.cell_size * self.map.shape[1],
			height=self.cell_size * self.map.shape[0]
		)
		self.canvas.pack()

		self.rectangles = [[None for _ in range(self.map.shape[1])] for _ in range(self.map.shape[0])]

		for y in range(self.map.shape[0]):
			for x in range(self.map.shape[1]):
				rect = self.canvas.create_rectangle(
					x * self.cell_size, y * self.cell_size,
					(x + 1) * self.cell_size, (y + 1) * self.cell_size,
					fill="white", outline="black"
				)
				self.rectangles[y][x] = rect

		self.root.update()
		self.update_map()
		self.root.mainloop()

	def update_map(self):
		color_map = {
			0: "white",
			'H': "blue",
			'S': "blue",
			'G': "green",
			'R': "red"
		}

		for y in range(self.map.shape[0]):
			for x in range(self.map.shape[1]):
				val = self.map[y, x]
				color = color_map.get(val, "gray")
				self.canvas.itemconfig(self.rectangles[y][x], fill=color)

		self.root.update()

	def place_items(self):
		for i in range (3):
			if i % 2 == 0:
				letter = 'G'
			else:
				letter = 'R'
			empty_positions = list(zip(*np.where(self.map == 0)))
			self.food_pos = empty_positions[np.random.randint(len(empty_positions))]
			self.map[self.food_pos] = letter

	def place_snake(self):
		snake_body = []
		empty_positions = list(zip(*np.where(self.map == 0)))
		head_pos = empty_positions[np.random.randint(len(empty_positions))]
		self.map[head_pos] = 'H'
		snake_body.append(head_pos)

		next = self.get_random_valid_neighbor(head_pos)
		self.map[next] = 'S'
		snake_body.append(next)
		next = self.get_random_valid_neighbor(next)
		self.map[next] = 'S'
		snake_body.append(next)
		self.snake = Snake(snake_body)

	def get_random_valid_neighbor(self, pos):
		height, width = self.map.shape
		y, x = pos
		neighbors = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
		valid = []
		for ny, nx in neighbors:
			if 0 <= ny < height and 0 <= nx < width:
				if self.map[ny, nx] == 0:
					valid.append((ny, nx))
		if valid:
			return valid[np.random.randint(len(valid))]
		return None

	def move(self, option):
		if option in ("UP", "DOWN", "LEFT", "RIGHT"):
			self.snake.move(option)
			self.update()

	
	def update(self):
		pass
		# new_position = self.snake.get_position()
		# head = new_position[0]
		# y, x = head
		# if 0 <= y < self.map.shape[0] and 0 <= x < self.map.shape[1]: