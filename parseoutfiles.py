import sys

import json

def getValueFromJsonFile(jsonfile,param):
	with open(jsonfile) as data_file:    
	    data = json.load(data_file)
	    return data[param]

def getOutputParam(param,filename='hypothesis_out.txt'):
	labels =  ["training_error","numberOfRules","test_accuracy","fitness",
				"training_accuracy","test_error","hyphotesis","size"]
	if not (param in labels):
		print 'param does not exist in the default inputs'
	return getValueFromJsonFile(filename,param)

def getInputParam(param,filename='input_params.txt'):
	labels = [
	"elitismRate", "mutationRate", "initrules", "decay", "maxrules", "selector",
	"populationSize", "crossoverRate","generations","dataset_file"]
	if not (param in labels):
		print 'param does not exist in the default inputs'
	return getValueFromJsonFile(filename,param)

"""
	For a given header label, returns the list of values relative to the label
	@param filename file to parse
	@label label to look up for in the header
	@return values relative to the label
"""
def getTrainingValues(label,filename='gabil-learning.txt'):
	labels=["#Generation","BestIndRuleSize","BestIndRawScore","BestIndAccuracy","BestIndError","WorstIndRuleSize",
			"WorstIndRawScore","WorstIndAccuracy","WorstIndError","AvgRuleSize","AvgRawScore","AvgAccuracy",
			"AvgError"]
	label_values = []
	with open(filename,'r+') as dataset:
		data = dataset.readlines()
		examples = []
		for entry in data:
			entry = entry[:-1] #remove last <whitespace> from each line 
			entries = entry.split(',')

			#case when we are parsing the labels, searching for the input label index
			if entry[0]=="#": 
				label_index = -1
				for index,value in enumerate(entries):
					if value==label:
						label_index = index
						break
				if label_index ==-1:
					print 'error: label not found, returning empty list'
					return []
			#parsing actual values
			else: 
				label_values.append(entries[label_index])


	return label_values

if __name__ == "__main__":
	
	"""
	#Example of use
	generations = getTrainingValues(label="BestIndRawScore",filename='gabil-learning.txt')
	print generations

	input_ = getInputParam(param="crossoverRate",filename='input_params.txt')
	print input_

	output_ = getOutputParam(param="hyphotesis",filename='hypothesis_out.txt')
	print output_
	"""
