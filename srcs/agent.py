import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from data_utils import processPOV


gamma = 0.99

class Agent(nn.Module):
	def __init__(self):
		super(Agent, self).__init__()
		self.epsilon = 1.0
		self.epsilon_min = 0.01     # Valeur minimale (toujours un peu d’exploration)
		self.epsilon_decay = 0.995
		self.last_input = None
		self.last_action = None
		self.reward = 0
		self.Qvalue = 0
		# Couche 1 : entrée -> 128 neurones
		self.fc1 = nn.Linear(15, 64)
		# Couche 2 : 128 -> 64 neurones
		self.fc2 = nn.Linear(64, 64)
		# Couche de sortie : 64 -> nombre d'actions
		self.out = nn.Linear(64, 3)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

	def getDirection(self, AgentPOV, ActualDirection):
		if random.random() < self.epsilon:
			pred_index = random.randint(0, 2)
			self.last_input = (AgentPOV, ActualDirection)
		else:
			output = self.predict(AgentPOV, ActualDirection)
			self.last_input = (AgentPOV, ActualDirection)
			pred_index = torch.argmax(output).item()
			self.last_action = pred_index
			self.Qvalue = output[0][pred_index]
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		return pred_index
	
	def predict(self, AgentPOV, ActualDirection):
		tensor_input = processPOV(AgentPOV, ActualDirection)
		tensor_input = F.relu(self.fc1(tensor_input))
		tensor_input = F.relu(self.fc2(tensor_input))
		output = self.out(tensor_input)
		return output
	
	def feedback(self, feedback):
		rewards = {
			"GameOver": -1,
			"Default": -0.05,
			"RedApple": -0.7,
			"GreenApple": 0.7,
			"Win": 1
		}
		self.reward = rewards.get(feedback, 0)
		# print(f"Feedback: {feedback}")
	
	def learn(self, newPOV, newDirection):
		AgentPOV, ActualDirection = self.last_input
		output = self.predict(AgentPOV, ActualDirection)
		Qvalue = output[0][self.last_action]

		if self.reward == -1:
			Qtarget = self.reward
		else:
			with torch.no_grad():
				next_q_values = self.predict(newPOV, newDirection)
				max_next_q = torch.max(next_q_values).item()
			Qtarget = self.reward + gamma * max_next_q

		loss = ((Qvalue - Qtarget)**2).mean()
		# print(f"LOSS = {loss}")

		self.optimizer.zero_grad()  # remise à zéro des gradients
		loss.backward()        # calcul des gradients
		self.optimizer.step()       # mise à jour des poids
		return loss.item()