from SnakeGame import SnakeGame
import tkinter as tk
from agent import Agent
import argparse

parser = argparse.ArgumentParser(description="Learn2Sliter: train / demo")
subparsers = parser.add_subparsers(dest="action", required=True)

parser_train = subparsers.add_parser("train", help="Train the model")
parser_train.add_argument('--sessions', type=int, required=True, help="Number of training sessions")
parser_train.add_argument('--save', type=str, required=True, help='Path to save')
parser_train.add_argument('--visual', type=str, required=True, help='on or off')


parser_demo = subparsers.add_parser("demo", help="Demo with trained model")
parser_demo.add_argument('--load', type=str, required=True, help='Path of the model')

args = parser.parse_args()

if args.action == "train":
	agent = Agent()
	root = tk.Tk()
	root.title("Learn2slither")
	game = SnakeGame(root, agent, "train", args.sessions)
	root.mainloop()

elif args.action == "demo":
	print(args.load)
	agent = Agent() # Change to load Agent
	root = tk.Tk()
	root.title("Learn2slither")
	game = SnakeGame(root, agent, "demo", 1)
	root.mainloop()

# Launch the Snake Game

