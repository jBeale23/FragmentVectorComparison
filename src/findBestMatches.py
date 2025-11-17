import argparse
import os
import pathlib
import sys
from itertools import repeat

import numpy as np
from basicParallelize import parallelProcess, parallelProcessTQDM


def findBestMatchPerQueryVector(
    matchDir: pathlib.Path,
    outputFile: pathlib.Path,
    separator="\t",
):
    if separator == "\t":
        fileType = "*.tsv"
    elif separator == ",":
        fileType = "*.csv"
    else:
        fileType = "*.txt"
    if (FVCoutputs := list(matchDir.glob(fileType))) != []:
        bestMatchIndices: list[tuple[int, int]] = []
        bestCosineSimilarities: list[np.float32] = []
        for file in FVCoutputs:
            proteinOneIndices = []
            proteinTwoIndices = []
            cosineSimilarities = []
            with open(file, "r") as inFile:
                for line in inFile:
                    if line.startswith("#"):
                        continue
                    else:
                        fields = line.rstrip("\n").split(separator)
                        proteinOneIndices.append(int(fields[0]))
                        proteinTwoIndices.append(int(fields[1]))
                        cosineSimilarities.append(float(fields[2]))
            bestMatch = np.argmax(cosineSimilarities)
            bestCosineSimilarities.append(cosineSimilarities[bestMatch])
            bestMatchIndices.append(
                (proteinOneIndices[bestMatch], proteinTwoIndices[bestMatch])
            )

        bestCosineSimilarityIndex = np.argmax(bestCosineSimilarities)
        bestMatchFile = FVCoutputs[bestCosineSimilarityIndex]
        bestMatchIndex = bestMatchIndices[bestCosineSimilarityIndex]
        bestCosineSimilarity = bestCosineSimilarities[bestCosineSimilarityIndex]
        with open(bestMatchFile, mode="r") as file:
            windowSize = file.readline().rstrip("\n").replace("#Window Size: ", "")
            percentile = file.readline().rstrip("\n").replace("#Percentile: ", "")
        with open(outputFile, mode="a") as outFile:
            outFile.write(
                f"{bestMatchFile}{separator}{bestMatchIndex[0]}{separator}{bestMatchIndex[1]}{separator}{bestCosineSimilarity}{separator}{windowSize}{separator}{percentile}\n"
            )
    else:
        if matchDir.stem not in ["eScape", "tmp"]:
            print(f"No FVC output files found in {matchDir}", file=sys.stderr)


def main(
    inputDir: pathlib.Path,
    outputFile: pathlib.Path | None = None,
    separator="\t",
):
    if inputDir.exists():
        if inputDir.is_dir():
            comparisonDirList = list(inputDir.glob("*/"))
            if outputFile is None:
                outputFile = inputDir.joinpath("FVC.summary")
            if not outputFile.exists():
                outputFile.parent.mkdir(parents=True, exist_ok=True)
                with open(outputFile, mode="w") as outFile:
                    outFile.write(
                        f"BestMatchFile{separator}QueryProteinIndex{separator}TestProteinIndex{separator}CosineSimilarity{separator}WindowSize{separator}Percentile\n"
                    )
            queryVectorsToMatch: list[tuple[pathlib.Path, pathlib.Path, str]] = list(
                zip(comparisonDirList, repeat(outputFile), repeat(separator))
            )
            parallelismArgs = {
                "function": findBestMatchPerQueryVector,
                "args": queryVectorsToMatch,
            }
            if "FVC_PIPELINE" not in os.environ:
                parallelismArgs.setdefault("chunkSize", 1)
                parallelismArgs.setdefault(
                    "description",
                    "Finding Best Regions of Similarity for Query Protein Vectors...",
                )
                parallelism = parallelProcessTQDM
            else:
                parallelism = parallelProcess
            parallelism(**parallelismArgs)

        else:
            raise TypeError(f"{inputDir} is not a directory.")
    else:
        raise FileNotFoundError(f"Results directory not found: {inputDir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Finds the best region of similarity for each query protein in the given directory"
    )
    parser.add_argument(
        "QUERYPROTEINS",
        type=pathlib.Path,
        help="The directory of proteins to find the best region of similarity for.",
    )
    parser.add_argument(
        "-o",
        "--OUTPUTFILE",
        type=pathlib.Path,
        default=None,
        help="The file to output best regions of similarity to. Defaults to QUERYPROTEINS/FVC.summary.",
    )
    parser.add_argument(
        "-S",
        "--SEPARATOR",
        type=str,
        default="\t",
        help="The string used to separate values in output files. Defaults to tab.",
    )
    args = parser.parse_args()
    main(
        inputDir=args.QUERYPROTEINS,
        outputFile=args.OUTPUTFILE,
        separator=args.SEPARATOR,
    )
