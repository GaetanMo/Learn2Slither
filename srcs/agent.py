import torch
import torch.nn as nn
import torch.nn.functional as F

gamma = 0.99

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
		self.reward = 0
		self.Qvalue = 0
		# Couche 1 : entrée -> 128 neurones
		self.fc1 = nn.Linear(200, 64)
		# Couche 2 : 128 -> 64 neurones
		self.fc2 = nn.Linear(64, 64)
		# Couche de sortie : 64 -> nombre d'actions
		self.out = nn.Linear(64, 4)
		self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

	def getDirection(self, AgentPOV):
		output = self.predict(AgentPOV)
		print(output)
		pred_index = torch.argmax(output).item()
		max_value = output[0][pred_index] # Get Q-value
		self.Qvalue = max_value.item() # Store Q-value
		directions = ['Up', 'Down', 'Left', 'Right']
		pred_direction = directions[pred_index]
		print("Direction choisie :", pred_direction)
		return pred_direction

	def predict(self, AgentPOV):
		POV = []
		for head_y, row in enumerate(AgentPOV):
			if "H" in row:
				head_x = row.index("H")
				break
		# Get vision in each direction
		top = AgentPOV[0:head_y]
		bottom = AgentPOV[head_y + 1:]
		left = AgentPOV[head_y][0:head_x]
		right = AgentPOV[head_y][head_x + 1:len(AgentPOV[head_x])]

		print(top)

		for y, row in enumerate(top):
			for x, cell in enumerate(row):
				if cell != " ":
					POV.append(cell)
		while len(POV) < 10:
			POV.insert(0, "W")

		for y, row in enumerate(bottom):
			for x, cell in enumerate(row):
				if cell != " ":
					POV.append(cell)
		while len(POV) < 20:
			POV.append("W")
		while len(left) < 10:
			left.insert(0, "W")
		POV = POV + left + right
		while len(POV) < 40:
			POV.append("W")
		# print(POV)

		one_hot_list = [encoding[symbol] for symbol in POV]
		tensor_input = torch.tensor(one_hot_list, dtype=torch.float32).flatten()
		tensor_input = tensor_input.unsqueeze(0)
		tensor_input = F.relu(self.fc1(tensor_input))
		tensor_input = F.relu(self.fc2(tensor_input))
		output = self.out(tensor_input)
		return output
	
	def feedback(self, feedback):
		if feedback == "GameOver":
			print("Feedback: Game Over")
			self.reward = -10
		elif feedback == "Default":
			print("Feedback: Default")
			self.reward = -1
		elif feedback == "RedApple":
			print("Feedback: RedApple")
			self.reward = -5
		elif feedback == "GreenApple":
			print("Feedback: GreenApple")
			self.reward = 5
		elif feedback == "Win":
			print("Feedback: Win")
			self.reward = 10
	
	def learn(self, newPOV):
		output = self.predict(newPOV)
		pred_index = torch.argmax(output).item()
		max_value = output[0][pred_index] # Get Q-value
		Qfuture = max_value.item() # Store Q-value
		Qtarget = self.reward + gamma * Qfuture
		Qtarget = torch.tensor(Qtarget, dtype=torch.float32)
		loss = F.mse_loss(self.Qvalue, Qtarget)
		print(f"LOSS = {loss}")
		self.optimizer.zero_grad()  # remise à zéro des gradients
		loss.backward()        # calcul des gradients
		self.optimizer.step()       # mise à jour des poids
		return loss.item()