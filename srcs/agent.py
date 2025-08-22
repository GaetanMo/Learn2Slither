import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import processPOV


gamma = 0.99

class Agent(nn.Module):
	def __init__(self):
		super(Agent, self).__init__()
		self.reward = 0
		self.Qvalues = []
		self.last_action = "Up"
		# Couche 1 : entrée -> 128 neurones
		self.fc1 = nn.Linear(200, 64)
		# Couche 2 : 128 -> 64 neurones
		self.fc2 = nn.Linear(64, 64)
		# Couche de sortie : 64 -> nombre d'actions
		self.out = nn.Linear(64, 4)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

	def getDirection(self, AgentPOV):
		output = self.predict(AgentPOV)
		self.Qvalues = output # Store Q-value
		print(output)
		pred_index = torch.argmax(output).item()
		directions = ['Up', 'Down', 'Left', 'Right']
		pred_direction = directions[pred_index]
		print("Direction choisie :", pred_direction)
		self.last_action = pred_index
		return pred_direction

	def predict(self, AgentPOV):
		tensor_input = processPOV(AgentPOV)
		tensor_input = F.relu(self.fc1(tensor_input))
		tensor_input = F.relu(self.fc2(tensor_input))
		output = self.out(tensor_input)
		return output
	
	def feedback(self, feedback):
		rewards = {
			"GameOver": -100,
			"Default": -1,
			"RedApple": -50,
			"GreenApple": 50,
			"Win": 100
		}
		self.reward = rewards.get(feedback, 0)
		print(f"Feedback: {feedback}")
	
	def learn(self, newPOV):

		with torch.no_grad():
			next_q_values = self.predict(newPOV)
			max_next_q = torch.max(next_q_values).item()

		Qtarget = self.reward + gamma * max_next_q
		print(f"REWARD = {self.reward}")
		target_q_values = self.Qvalues.clone().detach()
		target_q_values[0][self.last_action] = Qtarget

		loss = F.mse_loss(self.Qvalues, target_q_values)
		print(f"LOSS = {loss.item()}")

		self.optimizer.zero_grad()  # remise à zéro des gradients
		loss.backward()        # calcul des gradients
		self.optimizer.step()       # mise à jour des poids
		return loss.item()