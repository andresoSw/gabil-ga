from pyevolve.GenomeBase import GenomeBase
from pyevolve.G1DBinaryString import G1DBinaryString
from pyevolve import Mutators
from pyevolve import Consts
from pyevolve import Util
from random import randint as rand_randint
import itertools

class GD1BinaryStringSet(GenomeBase):
	def __init__(self):
		GenomeBase.__init__(self)
		self.ruleSet = G1DBinaryString(0)
		self.rulePartition = []
		self.ruleSetSize = 0
		self.examples = []
		self.initializator.set(Consts.CDefG1DBinaryStringInit)
		self.mutator.set(WG1DBinaryStringSetMutatorFlip)
		self.crossover.set(G1DBinaryStringSetXTwoPoint)

	def __len__(self):
		return self.ruleSetSize

	def __getitem__(self, key):
		return self.ruleSet[key]

	def setExamplesRef(self,examples):
		self.examples = examples

	def getExamplesRef(self):
		return self.examples

	"""
		@return the distance from the left crossover cut to the start 
		of the closest rule by the left 
	"""
	def distanceFromCutToClosestRuleByLeft(self,cut):
		if ((cut < 0) or cut > self.ruleSetSize) :
			Util.raiseException("Crossover cut point %s is out of the bounds of the rule set <%s,%s>" %(cut,0,self.ruleSetSize), ValueError)
		shift = 0
		for lower,upper in self.rulePartition:
			if upper > cut: 
				return cut - lower

	def getCutPointsFromDistances(self,leftDistance,rightDistance):
		if ((leftDistance < 0) or (rightDistance < 0)) :
			Util.raiseException("leftDistance and rightDistance must be positive", ValueError)
		if (rightDistance > self.ruleSetSize):
			Util.raiseException("rightDistance is out of the bounds of the rule set size", ValueError)
		rightCutCandidates = [lower+leftDistance for (lower,_) in self.rulePartition]
		leftCutCandidates = [lower+rightDistance for (lower,_) in self.rulePartition]
		cross_product = itertools.product(rightCutCandidates,leftCutCandidates)
		return [(lower,upper) for (lower,upper) in list(cross_product) if lower<=upper]

	def recomputePartitions(self):
		lower,upper = self.rulePartition[0] #are rules are the same length, one can pick any rule
		rule_len = upper-lower
		if ((self.ruleSetSize % rule_len)!= 0) :
			Util.raiseException("rule set size must be module of rule size %s" %(rule_len), ValueError)
		self.rulePartition  = [(lower,lower+rule_len) for lower in range(0,self.ruleSetSize,rule_len) ]

	def substitute(self,leftCut,rightCut,subRule):
		if ((rightCut < 0) or (leftCut < 0)) :
			Util.raiseException("Crossover cut points must be positive", ValueError)
		if (leftCut > rightCut ):
			Util.raiseException("left cut must be lower than the right cut", ValueError)
		if (rightCut > self.ruleSetSize):
			Util.raiseException("rightCut is out of the bounds of the rule set size", ValueError)
		self.ruleSet[leftCut:rightCut] = subRule
		self.ruleSetSize = len(self.ruleSet)
		self.recomputePartitions()
	
	def addRule(self,rule):
		if not isinstance(rule,G1DBinaryString):
			Util.raiseException("The rule must of type G1DBinaryString", ValueError)
		rule_len = len(rule)
		newRuleset = G1DBinaryString(self.ruleSetSize + rule_len)
		newSet = []
		newSet.extend(self.ruleSet)
		newSet.extend(rule)
		for bit in newSet:
			newRuleset.append(bit)
		self.ruleSet = newRuleset
		self.rulePartition.append((self.ruleSetSize,self.ruleSetSize+rule_len))
		self.ruleSetSize = len(self.ruleSet)

	def addRuleAsString(self,ruleStr):
		if not isinstance(ruleStr,str):
			Util.raiseException("The rule must of type str", ValueError)
		rule = G1DBinaryString(len(ruleStr))
		for bit in ruleStr:
			rule.append(int(bit))
		self.addRule(rule)

	def getDecimal(self):
		return int(self.getBinary(), 2)

	def getBinary(self):
		return "".join(map(str, self.ruleSet))

	def G1DBinaryStringSetMutatorFlip(self,mutationRatio):
		Mutators.G1DBinaryStringMutatorFlip(self.ruleSet,pmut=mutationRatio)

	"""
		@param rule can be either a bitString in string format or a G1DBinaryString
		@return True if a given rule exists within the rule Set, False otherwise
	"""
	def ruleExists(self,rule):
		if not (isinstance(rule,str) or isinstance(rule,G1DBinaryString)):
			Util.raiseException("BitString expected as input", ValueError)
		if isinstance(rule,G1DBinaryString): rule = G1DBinaryString.getBinary()
		for lowerCut,upperCut in self.rulePartition:
			currentRule = ''.join(map(str,self.ruleSet[lowerCut:upperCut]))
			if (currentRule[:-1]==rule): return True
		return False

	"""
		@param rule can be either a bitString in string format or a G1DBinaryString
		@return The bit of the classification corresponding to the given rule, if more than
				one rule matches the given rule, the first that is found is retrieved by default
	"""
	def getClassificationForRule(self,rule):
		if not (isinstance(rule,str) or isinstance(rule,G1DBinaryString)):
			Util.raiseException("BitString expected as input", ValueError)
		if isinstance(rule,G1DBinaryString): rule = G1DBinaryString.getBinary()
		for lowerCut,upperCut in self.rulePartition:
			fullRule = ''.join(map(str,self.ruleSet[lowerCut:upperCut]))
			#current rule is obtained by ignoring the last classification bit
			currentRule = fullRule[:-1]
			if (currentRule==rule): return fullRule[-1] #last bit corresponds to the classification
		return None


	def __repr__(self):
		""" Return a string representation of Genome """
		ret = GenomeBase.__repr__(self)
		ret += "- G1DBinaryStringSet\n"
		ret += "\tString length:\t %s\n" % (self.ruleSetSize,)
		ret += "\tString:\t\t %s\n" % (self.getBinary(),)
		ret += "\tRule Partitions: %s\n\n" %(self.rulePartition,)
		return ret

	def copy(self, g):
		""" Copy genome to 'g' """
		GenomeBase.copy(self, g)
		g.ruleSet = self.ruleSet[:] #deep copy
		g.rulePartition = self.rulePartition[:] #deep copy
		g.ruleSetSize = self.ruleSetSize
		g.examples = self.examples # only a ref to the examples is copied

	def clone(self):
		""" Return a new instace copy of the genome """
		newcopy = GD1BinaryStringSet()
		self.copy(newcopy)
		return newcopy

"""
	Mutator method, wrapper for G1DBinaryStringMutatorFlip method
"""
def WG1DBinaryStringSetMutatorFlip(genome,**args):
	genome.G1DBinaryStringSetMutatorFlip(mutationRatio=args['pmut'])

"""
	Crossover method, adaptation of pyevolve G1DBinaryStringXTwoPoint function
"""
def G1DBinaryStringSetXTwoPoint(genome, **args):
   """The 1D Binary String Set crossover, Two Point"""

   sister = None
   brother = None
   gMom = args["mom"]
   gDad = args["dad"]
   
   if len(gMom) == 1:
      Util.raiseException("The Binary String have one element, can't use the Two Point Crossover method !", TypeError)

   #Generating random crossover points over the mother parent
   cutsMom = [rand_randint(1, len(gMom)-1), rand_randint(1, len(gMom)-1)]
   if cutsMom[0] > cutsMom[1]:
      Util.listSwapElement(cutsMom, 0, 1)

   # Computing the distance from the cuts to the nearest rule to the left
   cutsMomDistance = [gMom.distanceFromCutToClosestRuleByLeft(cutsMom[0]),
   						gMom.distanceFromCutToClosestRuleByLeft(cutsMom[1])]

   # Computing factible crossover points for the dad parent
   factibleDadCuts = gDad.getCutPointsFromDistances(cutsMomDistance[0],cutsMomDistance[1])
   #picking one random cut pair for the parent from the factible cuts
   cutsDad = factibleDadCuts[rand_randint(0,len(factibleDadCuts)-1)]

   # genome crossover
   if args["count"] >= 1:
      sister = gMom.clone()
      sister.resetStats()
      sister.substitute(cutsMom[0],cutsMom[1],gDad[cutsDad[0]:cutsDad[1]])

   if args["count"] == 2:
      brother = gDad.clone()
      brother.resetStats()
      brother.substitute(cutsDad[0],cutsDad[1], gMom[cutsMom[0]:cutsMom[1]])

   return (sister, brother)


"""
	fitness function
"""
def rule_eval(genome):
	examples = genome.getExamplesRef()
	if not isinstance(genome,GD1BinaryStringSet):
			Util.raiseException("The rule must of type G1DBinaryString", ValueError)
	corrects = 0.0
	last_elem_index = -1

	for example in examples:
		rule = example[:last_elem_index]
		classification = example[last_elem_index]
		if (genome.ruleExists(rule) and genome.getClassificationForRule(rule)==classification):
			corrects += 1

	#the final score is the classification accuracy to the power of 2
	score = (corrects/len(examples))**2
	return score

if __name__ == '__main__':
	"""
		Testing creation of rules and rule sets
	"""
	rule_length = 5 #bits

	#h1
	genomeh1 = GD1BinaryStringSet()

	#rule1
	rule1 = G1DBinaryString(rule_length)
	rule1.append(1)
	rule1.append(0)
	rule1.append(0)
	rule1.append(1)
	rule1.append(1)
	genomeh1.addRule(rule1)

	#
	rule2 = G1DBinaryString(rule_length)
	rule2.append(1)
	rule2.append(1)
	rule2.append(1)
	rule2.append(0)
	rule2.append(0)
	genomeh1.addRule(rule2)

	print 'genome h1:',genomeh1

	#h2
	genomeh2 = GD1BinaryStringSet()

	#rule1
	rule1 = G1DBinaryString(rule_length)
	rule1.append(0)
	rule1.append(1)
	rule1.append(1)
	rule1.append(1)
	rule1.append(0)
	genomeh2.addRule(rule1)

	#
	rule2 = G1DBinaryString(rule_length)
	rule2.append(1)
	rule2.append(0)
	rule2.append(0)
	rule2.append(1)
	rule2.append(0)
	genomeh2.addRule(rule2)

	print 'genome h2:',genomeh2


	"""
		Testing crossover method
	"""
	print 'crossover'
	sister,brother = G1DBinaryStringSetXTwoPoint(None,mom=genomeh1,dad=genomeh2,count=2)
	print 'sister',sister,'brother',brother


	"""
		Testing mutation method
	"""
	print 'sister mutation'
	sister.G1DBinaryStringSetMutatorFlip(mutationRatio=0.1)
	print sister

	#testgnome h2
	genomeh2test = GD1BinaryStringSet()
	genomeh2test.addRuleAsString('01110')
	genomeh2test.addRuleAsString('10010')
	print 'are genomeh2 y genomeh2test string equals? ',genomeh2.ruleSet.getBinary()==genomeh2test.ruleSet.getBinary()

	"""
		Testing fitness methods
	"""
	rule1gh2 = '01110'
	rule2gh2 = '10010'
	print 'exists rule?',genomeh2.ruleExists(rule1gh2[:-1]) #expected True
	print 'exists rule?',genomeh2.ruleExists(rule2gh2[:-1]) #expected True
	print 'exists rule?',genomeh2.ruleExists('00000') #expected False

	bitc1 = genomeh2.getClassificationForRule(rule1gh2[:-1])
	bitc2 = genomeh2.getClassificationForRule(rule2gh2[:-1])
	bitc3 = genomeh2.getClassificationForRule('0000')
	print 'classification bit for rule %s: %s'%(rule1gh2[:-1],bitc1)
	print 'classification bit for rule %s: %s'%(rule2gh2[:-1],bitc2)
	print 'classification bit for rule 0000: %s'%(bitc3)

	"""
		Expected fitness 1
	"""
	exmplesgh2 = ['01110','10010']
	genomeh2.setExamplesRef(exmplesgh2)
	print 'fitness for genomeh2: ' , rule_eval(genomeh2)

	"""
		Expected fitness 0.25
	"""
	exmplesgh2 = ['01110','10011']
	genomeh2.setExamplesRef(exmplesgh2)
	print 'fitness for genomeh2: ' , rule_eval(genomeh2)