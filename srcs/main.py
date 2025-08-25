from SnakeGame import SnakeGame
import tkinter as tk
from agent import Agent
# Launch the Snake Game
test = Agent()
root = tk.Tk()
root.title("Learn2slither")
game = SnakeGame(root, test, "train", 2500)
game = SnakeGame(root, test, "demo", 2500)
root.mainloop()
