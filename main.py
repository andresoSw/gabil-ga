from os import listdir
from os.path import isfile,join
from gabil import  AttributeBitString,ContinuousAttributeBitString
from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
import pyevolve as pyevolve
import BinarystringSet as BinaryStringSet
from random import randint as rand_randint

# The step callback function, this function
# will be called every step (generation) of the GA evolution
def evolve_callback(ga_engine):
   generation = ga_engine.getCurrentGeneration()
   print "Current generation: %d" % (generation,)
   print ga_engine.getStatistics()
   return False

def population_init(genome,**args):
	genomeExamples = genome.getExamplesRef()
	genome.addRuleAsString(genomeExamples[rand_randint(0,len(genomeExamples)-1)])

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

	#must be ordered
	binaryStreams = [A1BinaryStream,A2BinaryStream,A3BinaryStream,A4BinaryStream,A5BinaryStream,A6BinaryStream,
					A7BinaryStream,A8BinaryStream,A9BinaryStream,A10BinaryStream,A11BinaryStream,A12BinaryStream,
					A13BinaryStream,A14BinaryStream,A15BinaryStream,ClassificationBinaryStream]

	for binaryStream in binaryStreams:
		binaryStream.computeBinaryStream()

	ClassificationBinaryStream.getStream()['+'] = '1'
	ClassificationBinaryStream.getStream()['-'] = '0'

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
			examples = []
			for entry in data:
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
				examples.append(rule)

			"""
				Evolutionary Algorithm using pyevolve
			"""
			rule_length = len(examples[0])

			# Genome instance
			genome = BinaryStringSet.GD1BinaryStringSet(rule_length)
			genome.setExamplesRef(examples)

			# The evaluator function (fitness function)
			genome.evaluator.set(BinaryStringSet.rule_eval)
			genome.initializator.set(population_init)
			genome.mutator.set(BinaryStringSet.WG1DBinaryStringSetMutatorFlip)
			genome.crossover.set(BinaryStringSet.G1DBinaryStringSetXTwoPoint)
			# Genetic Algorithm Instance
			ga = GSimpleGA.GSimpleGA(genome)

			# Set the Roulette Wheel selector method, the number of generations and
			# the termination criteria
			ga.selector.set(Selectors.GRouletteWheel)
			ga.setCrossoverRate(1.0)
			ga.setGenerations(70)
			ga.setMutationRate(0.01)
			ga.setPopulationSize(10)

			# to be executed at each generation
			ga.stepCallback.set(evolve_callback)

			# Do the evolution
			ga.evolve()

			# Best individual
			print 'Best individual:',ga.bestIndividual()

if __name__ == '__main__':
	datasets_dir = 'datasets'
	trainDatasetsInDir(datasets_dir)
