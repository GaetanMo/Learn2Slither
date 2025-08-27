from SnakeGame import SnakeGame
import tkinter as tk
from agent import Agent
import argparse
import pickle

parser = argparse.ArgumentParser(description="Learn2Sliter: train / demo")
subparsers = parser.add_subparsers(dest="action", required=True)

parser_train = subparsers.add_parser("train", help="Train the model")
parser_train.add_argument('--sessions', type=int, required=True, help="Number of training sessions")
parser_train.add_argument('--save', type=str, required=True, help='Path to save')
parser_train.add_argument('--visual', type=str, required=True, help='on or off')
# parser_train.add_argument('--step-by-step', type=str, required=False, help='on or off')


parser_demo = subparsers.add_parser("demo", help="Demo with trained model")
parser_demo.add_argument('--sessions', type=int, required=True, help="Number of demo sessions")
parser_demo.add_argument('--load', type=str, required=True, help='Path of the model')
# parser_demo.add_argument('--step-by-step', type=str, required=False, help='on or off')

args = parser.parse_args()

if args.action == "train":
	agent = Agent()
	root = tk.Tk()
	root.title("Learn2slither")
	game = SnakeGame(root, agent, "train", args.sessions)
	root.mainloop()
	with open(args.save, "wb") as f:
		pickle.dump(agent, f)

elif args.action == "demo":
	try:
		with open(args.load, "rb") as f:
			agent = pickle.load(f)
		root = tk.Tk()
		root.title("Learn2slither")
		game = SnakeGame(root, agent, "demo", args.sessions)
		root.mainloop()
	except:
		print("File .pkl error !")


# Launch the Snake Game

