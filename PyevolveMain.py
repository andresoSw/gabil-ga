from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
import pyevolve as pyevolve
import BinarystringSet as BinaryStringSet

# The step callback function, this function
# will be called every step (generation) of the GA evolution
def evolve_callback(ga_engine):
   generation = ga_engine.getCurrentGeneration()
   print "Current generation: %d" % (generation,)
   print ga_engine.getStatistics()
   return False

def run_main():
   examples = ['01110','10011']
   rule_length = 5

   # Genome instance
   genome = BinaryStringSet.GD1BinaryStringSet(rule_length)
   genome.setExamplesRef(examples)

   # The evaluator function (fitness function)
   genome.evaluator.set(BinaryStringSet.rule_eval)
   genome.initializator.set(BinaryStringSet.GD1BinaryStringSetInitializator)
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

   # to be executed at each generation
   ga.stepCallback.set(evolve_callback)

    # Do the evolution
   ga.evolve()

   # Best individual
   print 'Best individual:',ga.bestIndividual()

if __name__ == "__main__":
   run_main()
