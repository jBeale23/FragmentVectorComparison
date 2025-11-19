# Fragment Vector Comparison (FVC) User Guide

## Getting Started
0. If you are on MacOS, install gcc version 15 from Homebrew with `brew install gcc@15` and OpenMP with `brew install libomp` if you want to use the C implementation of FVC. Editing the Makefile to specifically use the installed gcc-15 may be necessary due to Apple's default aliasing of gcc. For Linux distributions, the native gcc installation is usually sufficient, though you may need to install libomp through your distribution's package manager. Windows is not currently supported by the C implementation of FVC.
1. Run `./setup.sh` to automatically create the required conda environment.
2. Aquire one or more fasta sequences for one or more organisms from a database that follows Uniprot header conventions.
    - These can contain any arbitrary number of proteins in each fasta file, provided that there's few enough that they fit in the RAM of your computer.
    - Alternatively, place any number of fasta files containing only a single fasta sequence in one directory per group of files and use them as input for the pipeline.

## Automatically Finding Putative Sites of Cross Reactivity
This is the best option if you want to run the full analysis pipeline in without major alterations to the workflow.
1. Follow the steps in **Getting Started**.
2. Run `./FVC YourFastaFile YourSecondFastaFile` wherever you'd like the output files to be.
    - The inputs for this can be any fasta files or directories of fasta files, provided any fasta files containing multiple sequences conform to the Uniprot header conventions.
    - If you're running a particularly large comparison set, you can run FVC in a detached screen and check in on the progress of the run by looking at the `FVC.log` file in the `FVCresults-timeStamp` directory.
    - The python implementation of FVC is used by default, as the C implementation currently does not support Windows computers.
    - Window size can be specified with `-w` or `--windowSize`. If unspecified it defaults to 10. It can also be set to the word `length` (case insensitive) to use the length of the shorter protein as the window size for each comparison.
    - The percentile above which to return matches can be specified with `-p` or `--percentile`. If unspecified it defaults to the 99.99th percentile.
    - If FVC is interrupted, it can be resumed from its last checkpoint by providing the `-r or --resumeRun` option with a `FVCresults-timestamp` directory as an additional argument. Its checkpoint state is determined from the `FVC.log` file.

## Manually Finding Putative Sites of Cross Reactivity
This is the best option if you want more precise control over the pipeline's operations or are experiencing difficulties with the standard workflow.
1. Follow the steps in **Getting Started**.
2. Run `conda activate FVC` to activate the conda environment.
3. For each fasta file, run `./src/fastaSplitter.py FastaFile -o YourOutputDirectory` to split the fasta sequences.
    - If your multiple fasta file doesn't conform to the Uniprot header conventions, you can edit the RegEx in `./src/fastaSplitter.py` to change how headers are detected.
    - If you already have a directory of fasta files, each of which contains a single sequence, you can skip this step.
4. For each directory of fasta files, run `perl ./src/eScape/eScape_predictor_REP.pl YourFastaOutputDiredctory YourEscapeOutputDirectory`
    - This can take a while if you have a substantial number of fasta files (~100k+)
5. For each comparison you want to make, run `python ./src/fragmentVectorComparison.py YourEscapeOutputDirectory YourSecondEscapeOutputDirectory -w YourDesiredWindowSize -o YourComparisonOutputDirectory` or if on MacOS or Linux `./src/cFVC YourFirstEscapeOutputDirectory YourSecondEscapeOutputDirectory -w YourDesiredWindowSize -o YourComparisonOutputDirectory`.
    - The `-w` option specifies the window size, generally a value in the 10-20 range yields results most comparable to eTFR.
7. Finally, to retrieve the best matches for each protein, run `./src/findBestMatches.py YourComparisonOutputDirectory`
