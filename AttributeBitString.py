class AttributeBitString:

	def __init__(self,attributes,dontCareKey='dontCareKey',nullKey='nullKey'):
		self.attributes = attributes
		self.binaryStream = {}
		self.dontCareKey = dontCareKey
		self.nullKey = nullKey

	def getDontCare(self):
		return self.binaryStream[self.dontCareKey]

	def getNullMatch(self):
		return self.binaryStream[self.nullKey]

	def getStreamForAttribute(self,attributeKey):
		if attributeKey in self.binaryStream:
			return self.binaryStream[attributeKey]
		return None

	def getStream(self):
		return self.binaryStream

	def setStream(self,attributes):
		self.attributes = attributes

	def computeBinaryStream(self):
		number_of_attributes = len(self.attributes)
		default_binary_stream = '0'*number_of_attributes
		dont_care = '1'*number_of_attributes
		#adding null bitstring and dont care
		self.binaryStream[self.nullKey] = default_binary_stream
		self.binaryStream[self.dontCareKey] = dont_care
		#adding attributes bitstring
		for index,attribute in enumerate(reversed(self.attributes)):
			default_stream = list(default_binary_stream)
			#turning on the bit corresponding to the attribute
			default_stream[index] = '1' 
			self.binaryStream[attribute] = "".join(default_stream)

	def printBinaryStream(self):
		print 'BinaryStream: for attributes: %s\nStream: %s' %(self.attributes,self.binaryStream) 

class ContinuousAttributeBitString(AttributeBitString):
	def __init__(self,attributes,dontCareKey='dontCareKey',nullKey='nullKey'):
		AttributeBitString.__init__(self,attributes=attributes,dontCareKey=dontCareKey,nullKey=nullKey)

	def getStreamForAttribute(self,attributeKey):
		for lowerBound,upperBound in self.attributes:
			if ((lowerBound <= attributeKey) and (attributeKey < upperBound)):
				return self.binaryStream[(lowerBound,upperBound)]


if __name__ == '__main__':

	# Discrete attributes test
	test_attributes = ['a','b','c','d']
	binaryStream = AttributeBitString(test_attributes)
	binaryStream.computeBinaryStream()
	binaryStream.printBinaryStream()
	a_stream = binaryStream.getStreamForAttribute('a')
	print '\n<a> stream:',a_stream,'\n\n'

	# Continuous attributes test
	test_attributes = [(10.00,20.00),(20.00,25.00),(25.00,30.00),(30.00,40.00),(40.00,85.00)]
	binaryStream = ContinuousAttributeBitString(test_attributes)
	binaryStream.computeBinaryStream()
	binaryStream.printBinaryStream()
	c_stream = binaryStream.getStreamForAttribute(27.83)
	print '\n<27.83> stream:',c_stream