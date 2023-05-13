# GenAlg: Artificial Intelligence-based Subcode Design
## Description:
1. The implementation based on the thesis "Artificial Intelligence-based Subcode Design".
2. For the genetic algorithm implementation and the current configuraions:
- Delta, block size, number of iteration, number of popultation, mutation rate and early stop can be set up manually.
- Initialization: initialize with varying density.
- Genetic process: new population is the combination of selection from original population, mutation results and crossover results.
- Crossover: reserve the same bits of parents, and random assign values to remaining bits.
- Mutation: flip random number of bits.
- Selection: select num_pop from the previous iteration as the new population set, based on the fitness score of the individual. It aims to maximize the fitness score.
3. With Hash Table to record scores that have already been calculated.
## Folder Structure:
1. The datasets folder contains the set of Codewords (64, 47) and Reduced Codewords (64, 47).
- "Codewords (64, 47)": analysis_cerr_n64_k47_CRC0_SNR425_NewRadio_wlast9.csv
- "Reduced Codewords (64, 47)": u-patterns_n64_k47_5Greliab_delta5_withoutZeros_withWeights.csv
2. The results folder contains the generated matrix H_delta.
- "GenAlg zero init, SmallDataset": H_delta=5_weighted-stop=30-T=23-r_mut=0.1-reduced.csv
- "GenAlg new init, LargeDataset": H_delta=5_weighted-stop=40-T=30-r_mut=0.1.csv
3. GenAlg.py
4. README.md
## Instructions:
1. Run the script by 
```
python GenAlg.py -p <file path> -d <delta> -n <number of iterations> -t <T> -e <early stop rate> -r <mutation rate>
```