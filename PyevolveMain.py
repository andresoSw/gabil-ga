from pyevolve import G1DBinaryString
from pyevolve import GSimpleGA
from pyevolve import Selectors
from pyevolve import Mutators
import pyevolve as pyevolve

# This function is the evaluation function, we want
# to give high score to more zero'ed chromosomes
def eval_func(chromosome):
   score = 0.0

   # iterate over the chromosome
   for value in chromosome:
      if value == 0:
         score += 0.1
      
   return score

def run_main():
   print pyevolve.__file__

   # Genome instance
   genome = G1DBinaryString.G1DBinaryString()

   # The evaluator function (objective function)
   genome.evaluator.set(eval_func)
   genome.mutator.set(Mutators.G1DBinaryStringMutatorFlip)

   # Genetic Algorithm Instance
   ga = GSimpleGA.GSimpleGA(genome)
   
   # Set the Roulette Wheel selector method, the number of generations and
   # the termination criteria
   ga.selector.set(Selectors.GRouletteWheel)
   ga.setGenerations(70)

   # Do the evolution, with stats dump
   # frequency of 10 generations
   ga.evolve(freq_stats=20)

   # Best individual
   print ga.bestIndividual()

if __name__ == "__main__":
   run_main()
   