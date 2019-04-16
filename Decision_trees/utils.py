import math

def find_key_to_maxval(d):
	"""
	Function to make decision at leaf node (find key corresponding to maximum value in dictionary )
	Input:
	d: input dictionary of label distribution
	Output:
	Majority vote from the label distribution (key corresponding to maximum value)
	"""
	max_val, arg_val = -1, -1
	for key, value in d.items():
		if value > max_val:
			arg_val = key
			max_val = value
	return arg_val

def find_entropy(arr):
    """
    Function to find entropy of input array
    Input:
    X: input column
    Output:
    Entropy corresponding to the input column and value distribution (Number of values corresponding to y=1 and y=0)
    """
    count_store = {}
    total = 0
    for i in arr:
        count_store[i] = count_store.get(i,0) + 1
        total += 1
    entropy = 0
    for value in count_store.values():
        # Entropy calculation assuming base of e, base of 2 can be specified in the math.log function 
        # Since, sklearn uses base of e, this function uses the default base to validate results against sklearn
        entropy -= (value/total)*math.log(value/total)
    return entropy, count_store


