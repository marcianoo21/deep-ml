def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	if len(a[0]) != len(b):
		return -1
	else:
		final_matrix = []
		for i in range(len(a)):
			final_matrix.append(sum([a[i][j] * b[j] for j in range(len(b))]))
	return final_matrix


	