from SnakeGame import SnakeGame
import tkinter as tk
from agent import Agent
import argparse
import pickle

parser = argparse.ArgumentParser(description="Learn2Sliter: train / demo")
subparsers = parser.add_subparsers(dest="action", required=True)

p_train = subparsers.add_parser("train", help="Train the model")
p_train.add_argument('--sessions', type=int, required=True)
p_train.add_argument('--save', type=str, required=True)
p_train.add_argument('--visual', type=str, required=True)
p_train.add_argument('--step-by-step', type=str, required=False)


p_demo = subparsers.add_parser("demo")
p_demo.add_argument('--sessions', type=int, required=True)
p_demo.add_argument('--load', type=str, required=True)
p_demo.add_argument('--step-by-step', type=str, required=False)

args = parser.parse_args()

sbs = args.step_by_step

if args.action == "train":
    agent = Agent()
    root = tk.Tk()
    root.title("Learn2slither")
    if args.step_by_step == "on" or args.step_by_step == "off":
        game = SnakeGame(root, agent, "train", args.sessions, args.visual, sbs)
    else:
        game = SnakeGame(root, agent, "train", args.sessions, args.visual)
    root.mainloop()
    with open(args.save, "wb") as f:
        pickle.dump(agent, f)

elif args.action == "demo":
    try:
        with open(args.load, "rb") as f:
            agent = pickle.load(f)
        root = tk.Tk()
        root.title("Learn2slither")
        if args.step_by_step == "on" or args.step_by_step == "off":
            game = SnakeGame(root, agent, "demo", args.sessions, None, sbs)
        else:
            game = SnakeGame(root, agent, "demo", args.sessions)
        root.mainloop()
    except Exception as e:
        print("❌ Error loading .pkl :", e)
