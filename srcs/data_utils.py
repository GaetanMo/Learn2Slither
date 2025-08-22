import torch

encoding = {
	'0': [1, 0, 0, 0, 0],
	'S': [0, 1, 0, 0, 0],
	'R': [0, 0, 0, 0, 1],
	'W': [0, 0, 0, 1, 0],
	'G': [0, 0, 1, 0, 0]
}

def processPOV(AgentPOV):
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
		return tensor_input