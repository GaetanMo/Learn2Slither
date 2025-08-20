import torch
import torch.nn as nn
import torch.nn.functional as F

encoding = {
	'0': [1, 0, 0, 0, 0],
	'S': [0, 1, 0, 0, 0],
	'R': [0, 0, 0, 0, 1],
	'W': [0, 0, 0, 1, 0],
	'G': [0, 0, 1, 0, 0]
}


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
		POV = []
		rows = [AgentPOV[i*12:(i+1)*12] for i in range(12)]

		print(POV)
		# encoded = []
		# for char in POV:
		# 	encoded.extend(encoding.get(char, [1, 0, 0, 0]))  # default to empty if inconnu

		# # Convertir en tenseur
		# x = torch.tensor(encoded, dtype=torch.float32).unsqueeze(0)  # shape (1, 200)

		# # Tu peux maintenant passer ça à ton modèle
		# output = model(x)

		