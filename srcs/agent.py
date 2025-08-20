import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
	def __init__(self):
		super(Agent, self).__init__()

		# Couche 1 : entrée -> 128 neurones
		self.fc1 = nn.Linear(200, 64)
		# Couche 2 : 128 -> 64 neurones
		self.fc2 = nn.Linear(64, 64)
		# Couche de sortie : 64 -> nombre d'actions
		self.out = nn.Linear(64, 4)

	def forward(self, x):
		x = F.relu(self.fc1(x))    # activation ReLU
		x = F.relu(self.fc2(x))    # ReLU encore
		return self.out(x)         # pas d’activation finale (Q-valeurs brutes)

	def getDirection(self, AgentPOV):
		print(AgentPOV)