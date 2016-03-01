class AttributeBinaryStream:

	def __init__(self,attributes):
		self.attributes = attributes
		self.binaryStream = {}

	def getStream(self):
		return self.binaryStream

	def setStream(self,attributes):
		self.attributes = attributes

	def computeBinaryStream(self):
		number_of_attributes = len(self.attributes)
		default_binary_stream = '0'*number_of_attributes
		for index,attribute in enumerate(reversed(self.attributes)):
			default_stream = list(default_binary_stream)
			#turning on the bit corresponding to the attribute
			default_stream[index] = '1' 
			self.binaryStream[attribute] = "".join(default_stream)

	def printBinaryStream(self):
		print 'BinaryStream: for attributes: %s\nStream: %s' %(self.attributes,self.binaryStream) 

if __name__ == '__main__':

	test_attributes = ['a','b','c','d']
	binaryStream = AttributeBinaryStream(test_attributes)
	binaryStream.computeBinaryStream()
	binaryStream.printBinaryStream()
