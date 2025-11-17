#!/usr/bin/env python3
import argparse
import os
import pathlib
from itertools import repeat
from typing import Callable, Literal

os.environ["KMP_WARNINGS"] = (
    "off"  # This is used to silence a deprecation warning in numba as its version is pinned
)

import numba as nb
import numpy as np
from basicParallelize import parallelProcess, parallelProcessTQDM
from numpy.typing import NDArray


def loadProteinVector(filename: pathlib.Path) -> NDArray[np.float32]:
    data = np.loadtxt(filename, dtype=np.float32, delimiter="\t", usecols=range(8))
    return data


@nb.njit(nogil=True, cache=True)
def cosineSimilarity(
    v1: NDArray[np.float32],
    v2: NDArray[np.float32],
) -> float:
    magnitudeV1 = np.linalg.norm(v1)
    magnitudeV2 = np.linalg.norm(v2)
    if len(v2) < len(v1):
        v1 = v1[: len(v2)]
    elif len(v1) < len(v2):
        v2 = v2[: len(v1)]
    return np.dot(v1, v2) / (magnitudeV1 * magnitudeV2)


@nb.njit(nogil=True, cache=True)
def slidingWindowPredictSimilarity(
    protein1: NDArray[np.float32],
    protein2: NDArray[np.float32],
    similarityThreshold: float = 99.99,
    windowSize: int = 10,
) -> NDArray[np.float32]:
    if similarityThreshold < 0 or similarityThreshold > 100:
        raise ValueError("Similarity Threshold must be between 0 and 100 percent.")
    if windowSize > len(protein1) or windowSize > len(protein2):
        pairwiseMatches: NDArray[np.float32] = np.zeros(
            (len(protein1), 3), dtype=np.float32
        )
    else:
        protein1Chunks = np.lib.stride_tricks.sliding_window_view(
            protein1, windowSize, axis=0
        )
        protein2Chunks = np.lib.stride_tricks.sliding_window_view(
            protein2, windowSize, axis=0
        )
        numProtein1Chunks = len(protein1Chunks)
        numProtein2Chunks = len(protein2Chunks)
        metricArray: NDArray[np.float32] = np.zeros(
            (numProtein1Chunks, numProtein2Chunks), dtype=np.float32
        )
        for i in range(numProtein1Chunks):
            for j in range(numProtein2Chunks):
                flatChunk1 = protein1Chunks[i].flatten()
                flatChunk2 = protein2Chunks[j].flatten()
                metricArray[i, j] = cosineSimilarity(flatChunk1, flatChunk2)
        threshold = np.percentile(metricArray, similarityThreshold)
        putativeHitIndices = np.argwhere(metricArray >= threshold)
        protein1RegionsOfInterest: NDArray[np.float32] = protein1Chunks[
            putativeHitIndices[:, 0]
        ]
        protein2RegionsOfInterest: NDArray[np.float32] = protein2Chunks[
            putativeHitIndices[:, 1]
        ]
        bestMetricValues: NDArray[np.float32] = np.zeros(
            len(putativeHitIndices), dtype=np.float32
        )
        for i in range(len(putativeHitIndices)):
            bestMetricValues[i] = metricArray[
                putativeHitIndices[i, 0], putativeHitIndices[i, 1]
            ]
        pairwiseMatches = np.zeros(
            (len(protein1RegionsOfInterest), 3), dtype=np.float32
        )
        for i in range(len(protein1Chunks)):
            for j in range(len(protein2Chunks)):
                for k in range(len(protein1RegionsOfInterest)):
                    if np.all(
                        protein1Chunks[i] == protein1RegionsOfInterest[k]
                    ) and np.all(protein2Chunks[j] == protein2RegionsOfInterest[k]):
                        pairwiseMatches[k] = np.array([i, j, bestMetricValues[k]])

        pairwiseMatches[:, :2] += 1
    return pairwiseMatches


def saveOutput(
    queryProteinName: str,
    comparisonProteinName: str,
    pairwiseMatchesArray: NDArray[np.int32],
    similarityMetrics: NDArray[np.float32],
    similarityThreshold: float,
    windowSize: int,
    outputFile: pathlib.Path,
    separator: str = "\t",
) -> None:
    with open(outputFile, "w") as file:
        file.write(
            f"#Window Size: {windowSize}\n#Percentile: {similarityThreshold}\n#{queryProteinName}SequencePosition{separator}{comparisonProteinName}SequencePosition{separator}CosineSimilarity\n"
        )
        for i in range(len(pairwiseMatchesArray)):
            file.write(
                f"{separator.join(np.char.mod('%s', pairwiseMatchesArray[i]))}{separator}{similarityMetrics[i]}\n"
            )


def memoryManager(
    protein: pathlib.Path,
    comparisonProtein: pathlib.Path,
    outputDir: pathlib.Path,
    proteinVector: NDArray[np.float32],
    comparisonProteinVector: NDArray[np.float32],
    separator: str = "\t",
    similarityThreshold: float = 99.99,
    windowSize: int | Literal["length"] | None = None,
    overwrite: bool = False,
) -> None:
    """Wraps the sliding window comparison to ensure that there are only ever at most `nJobs` comparisons held in memory at a time."""

    # Checks if the output file already exists before spending time performing calculations again.
    queryProteinName = protein.stem
    comparisonProteinName = comparisonProtein.stem
    if separator == "\t":
        formatType: str = "tsv"
    elif separator == ",":
        formatType = "csv"
    else:
        formatType = "txt"
    # If no window size was provided, use the length of the shorter vector
    if not isinstance(windowSize, int):
        windowSize = min(len(proteinVector), len(comparisonProteinVector))

    outputFile = outputDir.joinpath(
        queryProteinName,
        f"{comparisonProteinName}.{formatType}",
    )
    if not (outputFile.exists() and outputFile.is_file()) or overwrite is True:
        matchesArray: NDArray[np.float32] = slidingWindowPredictSimilarity(
            protein1=proteinVector,
            protein2=comparisonProteinVector,
            similarityThreshold=similarityThreshold,
            windowSize=windowSize,
        )
        outputFile.parent.mkdir(exist_ok=True, parents=True)
        saveOutput(
            queryProteinName=queryProteinName,
            comparisonProteinName=comparisonProteinName,
            pairwiseMatchesArray=matchesArray[:, :2].astype(np.int32),
            similarityMetrics=matchesArray[:, 2],
            similarityThreshold=similarityThreshold,
            windowSize=windowSize,
            outputFile=outputFile,
            separator=separator,
        )
    return


def main(
    queryProtein: pathlib.Path,
    testProtein: pathlib.Path,
    outputDir: pathlib.Path | None = None,
    separator: str = "\t",
    similarityThreshold: float = 99.99,
    windowSize: int | None = None,
    overwrite: bool = False,
) -> None:
    if queryProtein.exists():
        if queryProtein.is_dir():
            queryProteins: list[pathlib.Path] = list(
                queryProtein.glob("*.eScapev8_pred*")
            )
            if len(queryProteins) == 0:
                raise FileNotFoundError(
                    f"No Query Protein Files found in: {queryProtein}"
                )
            parallelismArgs = {
                "function": loadProteinVector,
                "args": queryProteins,
            }
            if "FVC_PIPELINE" not in os.environ:
                parallelismArgs.setdefault("chunkSize", 1)
                parallelismArgs.setdefault(
                    "description",
                    "Loading Query Protein Vectors...",
                )
                parallelism = parallelProcessTQDM
            else:
                parallelism = parallelProcess
            queryProteinVectors = parallelism(**parallelismArgs)
        if queryProtein.is_file():
            if queryProteinFileSuffix := queryProtein.suffix == ".eScapev8_pred*":
                queryProteins = [queryProtein]
                queryProteinVectors = [loadProteinVector(queryProtein)]
            else:
                raise TypeError(
                    f"Type {queryProteinFileSuffix} is not supported. Only .eScapev8_pred derived files are supported."
                )
    else:
        raise FileNotFoundError(f"Query Protein File not found: {queryProtein}")

    if testProtein.exists():
        if testProtein.is_dir():
            testProteins: list[pathlib.Path] = list(
                testProtein.glob("*.eScapev8_pred*")
            )
            if len(testProteins) == 0:
                raise FileNotFoundError(
                    f"No Test Protein Files found in: {testProtein}"
                )
            parallelismArgs = {
                "function": loadProteinVector,
                "args": testProteins,
            }
            if "FVC_PIPELINE" not in os.environ:
                parallelismArgs.setdefault("chunkSize", 1)
                parallelismArgs.setdefault(
                    "description",
                    "Loading Test Protein Vectors...",
                )
                parallelism = parallelProcessTQDM
            else:
                parallelism = parallelProcess
            testProteinVectors = parallelism(**parallelismArgs)
        if testProtein.is_file():
            if testProteinFileSuffix := testProtein.suffix == ".eScapev8_pred*":
                testProteins = [testProtein]
                testProteinVectors = [loadProteinVector(testProtein, separator)]
            else:
                raise TypeError(
                    f"Type {testProteinFileSuffix} is not supported. Only .eScapev8_pred derived files are supported."
                )
    else:
        raise FileNotFoundError(f"Test Protein File not found: {testProtein}")

    if outputDir is None:
        outputDir = pathlib.Path().cwd().joinpath("vectorComparisonOutput")

    for i, protein in enumerate(queryProteins):
        comparisonMatrix: list[
            tuple[
                pathlib.Path,
                pathlib.Path,
                pathlib.Path,
                NDArray[np.float32],
                NDArray[np.float32],
                str,
                Callable[..., float],
                Literal["min", "max"],
                float,
                int | None,
            ]
        ] = list(
            zip(
                repeat(protein),
                testProteins,
                repeat(outputDir),
                repeat(queryProteinVectors[i]),
                testProteinVectors,
                repeat(separator),
                repeat(similarityThreshold),
                repeat(windowSize),
                repeat(overwrite),
            )
        )
        parallelismArgs = {
            "function": memoryManager,
            "args": comparisonMatrix,
        }
        if "FVC_PIPELINE" not in os.environ:
            parallelismArgs.setdefault("chunkSize", 1)
            parallelismArgs.setdefault(
                "description",
                f"Query Protein {i + 1}/{len(queryProteins)}: Calculating Regions of Similarity...",
            )
            parallelism = parallelProcessTQDM
        else:
            parallelism = parallelProcess
        parallelism(**parallelismArgs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compares vector representations of proteins to determine regions of maximum similarity."
    )
    parser.add_argument(
        "QUERYPROTEIN",
        type=pathlib.Path,
        help="The protein or directory of proteins to compare to other protein(s).",
    )
    parser.add_argument(
        "TESTPROTEIN",
        type=pathlib.Path,
        help="The protein or directory of proteins to compare query protein(s) to.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        default=None,
        help="The directory to output found matches to. Defaults to ./vectorComparisonOutput.",
    )
    parser.add_argument(
        "-w",
        "--windowSize",
        help="The size of sliding window to pass over each protein vector. Defaults to the length of the shorter protein.",
    )
    parser.add_argument(
        "-s",
        "--separator",
        type=str,
        default="\t",
        help="The string used to separate values in output files. Defaults to tab.",
    )
    parser.add_argument(
        "-p",
        "--percentile",
        type=float,
        default=99.99,
        help="The percentile of similarity above which to return matches. Defaults to 99.99%%.",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="If present, allows already existing output files at the same path to be overwritten.",
    )
    args = parser.parse_args()

    main(
        queryProtein=args.QUERYPROTEIN,
        testProtein=args.TESTPROTEIN,
        outputDir=args.output,
        separator=args.separator,
        similarityThreshold=args.percentile,
        windowSize=args.windowSize,
        overwrite=args.overwrite,
    )
