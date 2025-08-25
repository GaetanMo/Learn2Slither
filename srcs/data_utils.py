import torch

def processPOV(AgentPOV, ActualDirection):
		for head_y, row in enumerate(AgentPOV):
			if "H" in row:
				head_x = row.index("H")
				break
		moving_up = 0
		moving_down = 0
		moving_left = 0
		moving_right = 0
		danger_straight = 0
		danger_right = 0
		danger_left = 0
		food_up = 0
		food_down = 0
		food_left = 0
		food_right = 0
		poison_up = 0
		poison_down = 0
		poison_left = 0
		poison_right = 0
		if ActualDirection == 0:
			moving_up = 1
		elif ActualDirection == 1:
			moving_right = 1
		elif ActualDirection == 2:
			moving_down = 1
		elif ActualDirection == 3:
			moving_left = 1

		straight = None
		right = None
		left = None
		if moving_up:
			straight = AgentPOV[head_y - 1][head_x]    # Straight
			right = AgentPOV[head_y][head_x + 1]    # Right
			left = AgentPOV[head_y][head_x - 1]    # Left
		elif moving_down:
			straight = AgentPOV[head_y + 1][head_x]	# Straight
			right = AgentPOV[head_y][head_x - 1]	# Right
			left = AgentPOV[head_y][head_x + 1]	# Left
		elif moving_left:
			straight = AgentPOV[head_y][head_x - 1]	# Straight
			right = AgentPOV[head_y - 1][head_x]	# Right
			left = AgentPOV[head_y + 1][head_x]		# Left
		elif moving_right:
			straight = AgentPOV[head_y][head_x + 1]	# Straight
			right = AgentPOV[head_y + 1][head_x]	# Right
			left = AgentPOV[head_y - 1][head_x]	# Left

		if straight == "S" or straight == "W":
			danger_straight = 1
		if right == "S" or right == "W":
			danger_right = 1
		if left == "S" or left == "W":
			danger_left = 1
		# print(ActualDirection)
		# print(danger_straight, danger_right, danger_left)
		for y in range(head_y - 1, -1, -1):
			if AgentPOV[y][head_x] == "G":
				food_up = 1
				break
			elif AgentPOV[y][head_x] == "R":
				poison_up = 1
				break
		for y in range(head_y + 1, len(AgentPOV)):
			if AgentPOV[y][head_x] == "G":
				food_down = 1
				break
			elif AgentPOV[y][head_x] == "R":
				poison_down = 1
				break
		for x in range(head_x - 1, -1, -1):
			if AgentPOV[head_y][x] == "G":
				food_left = 1
				break
			elif AgentPOV[head_y][x] == "R":
				poison_left = 1
				break
		for x in range(head_x + 1, len(AgentPOV[0])):
			if AgentPOV[head_y][x] == "G":
				food_right = 1
				break
			elif AgentPOV[head_y][x] == "R":
				poison_right = 1
				break

		data = [
			danger_straight,
			danger_right,
			danger_left,
			moving_up,
			moving_down,
			moving_left,
			moving_right,
			food_up,
			food_down,
			food_left,
			food_right,
			poison_up,
			poison_down,
			poison_left,
			poison_right
		]
		# print(data)
		tensor_input = torch.tensor(data, dtype=torch.float32)
		tensor_input = tensor_input.unsqueeze(0)
		return tensor_input