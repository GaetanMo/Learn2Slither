import tkinter as tk
import random
from agent import Agent

GRID_SIZE = 10        # 10x10
CELL_SIZE = 60        # pixels per cell
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE

# Directions (x, y)
DIRECTIONS = {
	"Up": (0, -1),
	"Down": (0, 1),
	"Left": (-1, 0),
	"Right": (1, 0)
}

class SnakeGame:
	def __init__(self, root, Agent=None):
		self.Agent = Agent
		self.AgentPOV = None
		self.root = root
		self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="lightgrey")
		self.canvas.pack()

		# Random initial direction
		self.direction = random.choice(list(DIRECTIONS.keys()))
		dx, dy = DIRECTIONS[self.direction]

		# Snake initialisation
		while True:
			head_x = random.randint(0, GRID_SIZE - 1)
			head_y = random.randint(0, GRID_SIZE - 1)
			valid = True
			snake = []
			for i in range(3):
				x = head_x - i * dx
				y = head_y - i * dy
				if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
					valid = False
					break
				snake.append((x, y))
			if valid:
				self.snake = snake
				break

		self.running = True
		self.food_red, self.food_green = self.place_special_foods()
		self.print_map()
		self.root.bind("<Key>", self.change_direction)
		self.game_loop()

	def place_special_foods(self):
		# Place red food and two green foods
		empty = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
				 if (x, y) not in self.snake]
		food_red = random.choice(empty)
		empty.remove(food_red)
		food_green = random.sample(empty, 2)
		return food_red, food_green

	def game_loop(self):
		if self.running:
			self.draw()

	def draw(self):
		self.canvas.delete("all")
		# Draw the grid
		for x in range(GRID_SIZE):
			for y in range(GRID_SIZE):
				self.canvas.create_rectangle(
					x * CELL_SIZE, y * CELL_SIZE,
					(x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
					fill="white", outline="gray"
				)
		# Draw the special food
		rx, ry = self.food_red
		self.draw_cell(rx, ry, "red")
		for gx, gy in self.food_green:
			self.draw_cell(gx, gy, "green")
		# Draw the snake
		for i, (x, y) in enumerate(self.snake):
			color = "blue" if i == 0 else "blue"
			self.draw_cell(x, y, color)
		# Game Over message
		if not self.running:
			self.canvas.create_text(
				WIDTH // 2, HEIGHT // 2,
				text="Game Over",
				font=("Arial", 24),
				fill="red"
			)

	def draw_cell(self, x, y, color):
		self.canvas.create_rectangle(
			x * CELL_SIZE, y * CELL_SIZE,
			(x + 1) * CELL_SIZE, (y + 1) * CELL_SIZE,
			fill=color
		)

	def move_snake(self):
		dx, dy = DIRECTIONS[self.direction]
		head_x, head_y = self.snake[0]
		new_head = (head_x + dx, head_y + dy)

		# Eat red food
		if new_head == self.food_red:
			if len(self.snake) == 1:
				self.snake.insert(0, new_head)
				self.game_over()
				return
			self.snake.insert(0, new_head)
			self.snake.pop()
			if len(self.snake) > 1:
				self.snake.pop()
			empty = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
					 if (x, y) not in self.snake and (x, y) not in self.food_green]
			self.food_red = random.choice(empty)
		# Eat green food
		elif new_head in self.food_green:
			self.snake.insert(0, new_head)
			idx = self.food_green.index(new_head)
			empty = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
					 if (x, y) not in self.snake and (x, y) != self.food_red and (x, y) not in self.food_green]
			self.food_green[idx] = random.choice(empty)
		else:
			self.snake.insert(0, new_head)
			self.snake.pop()

	def change_direction(self, event):
		new_dir = event.keysym
		Agent.getDirection(Agent, self.AgentPOV)
		if new_dir in DIRECTIONS:
			# Empêche le demi-tour
			opposite = {
				"Up": "Down", "Down": "Up",
				"Left": "Right", "Right": "Left"
			}
			if opposite[new_dir] != self.direction:
				self.direction = new_dir
				self.move_snake()
				self.check_collisions()
				self.draw()
				self.print_map()
				print(f"{new_dir}")

	def check_collisions(self):
		head = self.snake[0]
		x, y = head

		# Wall colision
		if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
			self.game_over()

		# Self-collision
		if head in self.snake[1:]:
			self.game_over()

	def place_food(self):
		empty = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
				 if (x, y) not in self.snake]
		return random.choice(empty)

	def game_over(self):
		self.running = False
		self.canvas.create_text(
			WIDTH // 2, HEIGHT // 2,
			text="Game Over",
			font=("Arial", 24),
			fill="red"
		)

	def print_map(self):
		size = GRID_SIZE
		grid = [["0" for _ in range(size)] for _ in range(size)]
		head_x, head_y = self.snake[0]

		rx, ry = self.food_red
		grid[ry][rx] = "R"
		for gx, gy in self.food_green:
			grid[gy][gx] = "G"
		for i, (x, y) in enumerate(self.snake):
			if i == 0:
				grid[y][x] = "H"
			else:
				grid[y][x] = "S"

		for y in range(-1, size + 1):
			self.AgentPOV = []
			for x in range(-1, size + 1):
				if (x == -1 or x == size or y == -1 or y == size) and (x == head_x or y == head_y):
					self.AgentPOV.append("W")
				elif x == head_x or y == head_y:
					if 0 <= x < size and 0 <= y < size:
						self.AgentPOV.append(grid[y][x])
					else:
						self.AgentPOV.append("W")
				else:
					self.AgentPOV.append(" ")
			self.AgentPov = " ".join(self.AgentPOV)
			print(self.AgentPov)
