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
import sys, getopt
import time 

def extract_commandline_params(argv):
   how_to_use_message = '$ Usage: \n\tShort Version: python test.py -c <rate> ' \
                        '-m <rate> -g <howmany> -p <howmany> -d <datasetfile>\n'\
                        '\tLong Version : python test.py --crossover <rate>' \
                        '--mutation <rate> --generations <howmany> --population <howmany> '\
                        '--dataset <datasetfile>\n'

   mandatory_args = [("-c","--crossover"),("-m","--mutation"),("-g","--generations"),("-p","--population"),("-d","dataset")]

   # checking that all mandatory arguments were provide within the command line
   for shortArg,longArg in mandatory_args:
      if ((not shortArg in argv) and (not longArg in argv)):
         print "\n$ Execution Error: Missing argument \"%s\"" %(longArg)
         print how_to_use_message
         sys.exit(2)
  
   try:
      opts, args = getopt.getopt(argv,'c:m:g:p:d:',['crossover=','mutation=','generations=','population=','dataset='])
   except getopt.GetoptError:
      print how_to_use_message
      sys.exit(2)

   parsed_arguments = {}

   for opt, arg in opts:
      if opt in ("-c","--crossover"):
         parsed_arguments["crossover"] = float(arg)
      elif opt in ("-m","--mutation"):
         parsed_arguments["mutation"] = float(arg)
      elif opt in ("-g","--generations"):
         parsed_arguments["generations"] = int(arg)
      elif opt in ("-p","--population"):
         parsed_arguments["population"] = int(arg)
      elif opt in ("-d","--dataset"):
         parsed_arguments["dataset"] = arg

   return parsed_arguments


def accuracy(genome):
	examples = genome.getExamplesRef()
	attribute_bits = [2, 5, 4, 4, 3, 14, 9, 4, 2, 2, 5, 2, 3, 3, 4]
	if not isinstance(genome, BinaryStringSet.GD1BinaryStringSet):
			Util.raiseException("The rule must of type G1DBinaryString", ValueError)
	
	if (sum(attribute_bits) != genome.rule_length -1 ):
		Util.raiseException("Example is not consistent with its attributes", ValueError)

	rule_binary = genome.getBinary()
	rule_length = genome.rule_length
	rule_list = [rule_binary[i:i+rule_length] for i in xrange(0,len(rule_binary),rule_length)]


	corrects = 0.0
	for example in examples:
		corrects +=  BinaryStringSet.match_example(example,rule_list, attribute_bits)

	#the final score is the classification accuracy to the power of 2
	return corrects/len(examples)

# The step callback function, this function
# will be called every step (generation) of the GA evolution
def evolve_callback(ga_engine):
   generation = ga_engine.getCurrentGeneration()
   best_individual = ga_engine.bestIndividual()
   # generacion, bestgenome.numberofrules, fitness, %classification
   print "%s,%s,%s,%s" %(generation,len(best_individual.rulePartition),best_individual.fitness,accuracy(best_individual))
   return False

def population_init(genome,**args):
	MAX_NUMBER_OF_RULES = 5 # genome will have 1 <= i <= MAX rules within the rule set
	genomeExamples = genome.getExamplesRef()

	number_of_rules_to_add = rand_randint(1,MAX_NUMBER_OF_RULES)
	for i in range(0,number_of_rules_to_add):
		genome.addRuleAsString(genomeExamples[rand_randint(0,len(genomeExamples)-1)])

def train_gabil(crossoverRate,mutationRate,populationSize,generations,dataset_file):
	print '----------------------------------------------------------------'
	print 'Running GABIL with parameters:'
	print 'crossoverRate:',crossoverRate,
	print 'mutationRate:',mutationRate,
	print 'populationSize:',populationSize,
	print 'generations:',generations,
	print 'dataset_file',dataset_file, 


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
	with open(dataset_file,'r+') as dataset:
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
		genome.evaluator.set(BinaryStringSet.rule_eval2)
		genome.initializator.set(population_init)
		genome.mutator.set(BinaryStringSet.WG1DBinaryStringSetMutatorFlip)
		genome.crossover.set(BinaryStringSet.G1DBinaryStringSetXTwoPoint)
		# Genetic Algorithm Instance
		ga = GSimpleGA.GSimpleGA(genome)

		# Set the Roulette Wheel selector method, the number of generations and
		# the termination criteria
		ga.selector.set(Selectors.GRouletteWheel)
		ga.setCrossoverRate(crossoverRate)
		ga.setGenerations(1000)
		ga.setMutationRate(mutationRate)
		ga.setPopulationSize(populationSize)

		# to be executed at each generation
		ga.stepCallback.set(evolve_callback)

		# Do the evolution
		ga.evolve()

		# Best individual
		print 'Best individual:',ga.bestIndividual()

if __name__ == '__main__':

	#ignoring the name of the program from the command line args
	arguments = extract_commandline_params(sys.argv[1:]) 
	crossoverRate = arguments["crossover"]
	mutationRate = arguments["mutation"]
	populationSize = arguments["population"]
	generations = arguments["generations"]
	dataset_file = arguments["dataset"]
	train_gabil(crossoverRate=crossoverRate,mutationRate=mutationRate,
						populationSize=populationSize,generations=generations,dataset_file=dataset_file)
