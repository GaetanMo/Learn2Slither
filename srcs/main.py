from SnakeGame import SnakeGame
import tkinter as tk
from agent import Agent
# Launch the Snake Game
test = Agent()
root = tk.Tk()
root.title("Learn2slither")
game = SnakeGame(root, test)
root.mainloop()
