from os import listdir
from os.path import isfile,join
from gabil import  AttributeBitString,ContinuousAttributeBitString


def trainDatasetsInDir(dataset_directory):

	"""
		Initializing attributes bitstrings
	"""
	# Continuous attributes
	A2BinaryStream = ContinuousAttributeBitString([(10.00,20.00),(20.00,25.00),(25.00,30.00),(30.00,40.00),(40.00,86.00)])
	A3BinaryStream = ContinuousAttributeBitString([(0.00,2.50), (2.50,5.00), (5.00,10.00), (10.00,31.00)])
	A8BinaryStream = ContinuousAttributeBitString([(0.00,0.50), (0.50,1.50), (1.50,5.00), (5.00,31.00)])
	A11BinaryStream = ContinuousAttributeBitString([(0.00,1.00), (1.00,3.00), (3.00,5.00), (5.00,10.00), (10.00,71.00)])
	A14BinaryStream = ContinuousAttributeBitString([(0.00,100.00), (100.00,300.00), (300.00,2001.00)])
	A15BinaryStream = ContinuousAttributeBitString([(0.00,100.00), (100.00,500.00), (500.00,1000.00), (1000.00,100001.00)])

	# Discrete attributes
	A1BinaryStream = AttributeBitString(['b','a'])
	A4BinaryStream = AttributeBitString(['u','y','l','t'])
	A5BinaryStream = AttributeBitString(['g','p','gg']) 
	A6BinaryStream = AttributeBitString(['c','d','cc','i','j','k','m','r','q','w','x','e','aa','ff'])
	A7BinaryStream = AttributeBitString(['v','h','bb','j','n','z','dd','ff','o'])
	A9BinaryStream = AttributeBitString(['t','f'])
	A10BinaryStream = AttributeBitString(['t','f']) 
	A12BinaryStream = AttributeBitString(['t','f'])
	A13BinaryStream = AttributeBitString(['g','p','s'])
	ClassificationBinaryStream = AttributeBitString(['+','-'])
	ClassificationBinaryStream.getStream()['+'] = '1'
	ClassificationBinaryStream.getStream()['-'] = '0'

	#must be ordered
	binaryStreams = [A1BinaryStream,A2BinaryStream,A3BinaryStream,A4BinaryStream,A5BinaryStream,A6BinaryStream,
					A7BinaryStream,A8BinaryStream,A9BinaryStream,A10BinaryStream,A11BinaryStream,A12BinaryStream,
					A13BinaryStream,A14BinaryStream,A15BinaryStream,ClassificationBinaryStream]

	for binaryStream in binaryStreams:
		binaryStream.computeBinaryStream()

	"""
		Parsing Dataset
	"""
	for index,file_name in enumerate(listdir(dataset_directory)):
		print "%s)=============================================================" %(index+1)
		print "Training network from dataset: %s" %(file_name)
		data_file = join(dataset_directory,file_name)
		if not isfile(data_file): continue

		with open(data_file,'r+') as dataset:
			data = dataset.readlines()
			for linenum,entry in enumerate(data):
				rule = ""
				entry = entry[:-1] #remove last <whitespace> from each line 
				entries = entry.split(',')
				assert(len(entries)==len(binaryStreams))
				""" 
					Assumes entries are ordered with respect of the attributes
					Rule is extended with the bitstring representation of every attribute
				"""
				for index,attributeValue in enumerate(entries):
					#unknown attributes are valued as dont care bitstrings
					if attributeValue == '?':
						rule += binaryStreams[index].getDontCare()
						continue
					#casting to float needed for continuous attributes
					if isinstance(binaryStreams[index],ContinuousAttributeBitString):
						attributeValue = float(attributeValue)
					attributeAsBitString = binaryStreams[index].getStreamForAttribute(attributeValue)
					rule += attributeAsBitString
				print "%s:%s" %(linenum,rule)


if __name__ == '__main__':
	datasets_dir = 'datasets'
	trainDatasetsInDir(datasets_dir)
