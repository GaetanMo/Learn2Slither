import tkinter as tk
import random
import numpy as np
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
		self.AgentPOV = []
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
		self.getPOV()
		self.root.bind("<Key>", self.change_direction)
		self.game_loop()

	def game_loop(self):
		if self.running:
			self.draw()

	def place_special_foods(self):
		# Place red food and two green foods
		empty = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
				 if (x, y) not in self.snake]
		food_red = random.choice(empty)
		empty.remove(food_red)
		food_green = random.sample(empty, 2)
		return food_red, food_green

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
			self.Agent.feedback("RedApple")
		# Eat green food
		elif new_head in self.food_green:
			self.snake.insert(0, new_head)
			idx = self.food_green.index(new_head)
			empty = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
					 if (x, y) not in self.snake and (x, y) != self.food_red and (x, y) not in self.food_green]
			self.food_green[idx] = random.choice(empty)
			self.Agent.feedback("GreenApple")
		else: # Move normally
			self.snake.insert(0, new_head)
			self.snake.pop()
			self.Agent.feedback("Default")


	def change_direction(self, event):
		new_dir = self.Agent.getDirection(self.AgentPOV)
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
				self.getPOV()
				self.Agent.learn(self.AgentPOV)
				print(self.AgentPov)
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
		self.Agent.feedback("GameOver")
		self.running = False
		self.canvas.create_text(
			WIDTH // 2, HEIGHT // 2,
			text="Game Over",
			font=("Arial", 24),
			fill="red"
		)

	def getPOV(self):
		size = GRID_SIZE
		grid = [["0" for _ in range(size)] for _ in range(size)]

		grid[self.food_red[1]][self.food_red[0]] = "R"
		for gx, gy in self.food_green:
			grid[gy][gx] = "G"

		for i, (x, y) in enumerate(self.snake):
			grid[y][x] = "H" if i == 0 else "S"
		head_x, head_y = self.snake[0]
		self.AgentPOV = []

		for y in range(-1, size + 1):
			row = []
			for x in range(-1, size + 1):
				if x == head_x or y == head_y:
					if 0 <= x < size and 0 <= y < size:
						row.append(grid[y][x])
					else:
						row.append("W")
				else:
					row.append(" ")
			self.AgentPOV.append(row)
		self.AgentPov = "\n".join(" ".join(row) for row in self.AgentPOV)

