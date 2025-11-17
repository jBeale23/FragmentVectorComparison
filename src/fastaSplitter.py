#!/usr/bin/env python3
import argparse
import os
import pathlib
from itertools import repeat
from re import findall

from basicParallelize import parallelProcess, parallelProcessTQDM


def retrieveSequences(filename: pathlib.Path) -> tuple[list[str], list[str]]:
    """Uses regex to find all sequences within a fasta file."""
    with open(filename, "r") as inFile:
        allSequences = inFile.read()
        # NOTE: If you are using fasta files retrieved from a database not following uniprot format conventions
        # NOTE: the regex for identifying headers and the start of each fasta will need to be adjusted
        # NOTE: The error raised if no headers are found will hopefully always alert you if this is the case
        headers: list[str]
        if (headers := findall(r"(?<=>[ts][rp]\|).*(?=\|)", allSequences)) != []:
            fastas: list[str] = findall(r">[ts][rp]\|.*\|.*\n[A-Z\n]+", allSequences)
        else:
            raise ValueError(
                f"No valid fasta headers were found in: {filename}\nConsider adjusting the regex used for identifying sequences."
            )
    return headers, fastas


def writeFastas(
    outputDir: pathlib.Path,
    baseFileName: str,
    fasta: str,
    overwrite: bool = False,
) -> None:
    fileName = pathlib.Path(baseFileName).with_suffix(".fasta")
    outputFile = outputDir.joinpath(fileName)
    if not (outputFile.exists() and outputFile.is_file()) or overwrite is True:
        outputFile.parent.mkdir(exist_ok=True, parents=True)
        with open(outputFile, "w") as outFile:
            outFile.write(f"{fasta}\n")


def main(
    inputFasta: pathlib.Path,
    outputDir: pathlib.Path | None = None,
    overwrite: bool = False,
) -> None:
    if inputFasta.exists() and inputFasta.is_file():
        if fileSuffix := inputFasta.suffix == ".fasta":
            outputFiles, fastas = retrieveSequences(inputFasta)
            if outputDir is None:
                outputDir = pathlib.Path().cwd().joinpath("splitFastas")
            fastasToWrite: list[tuple[pathlib.Path, str, str, bool]] = list(
                zip(
                    repeat(outputDir),
                    outputFiles,
                    fastas,
                    repeat(overwrite),
                )
            )
            parallelismArgs = {
                "function": writeFastas,
                "args": fastasToWrite,
            }
            if "FVC_PIPELINE" not in os.environ:
                parallelismArgs.setdefault("chunkSize", 1)
                parallelismArgs.setdefault(
                    "description",
                    "Writing Split Fasta Files...",
                )
                parallelism = parallelProcessTQDM
            else:
                parallelism = parallelProcess
            parallelism(**parallelismArgs)
            if parallelism == parallelProcess:
                parallelism(
                    function=writeFastas,
                    args=fastasToWrite,
                    chunkSize=1,
                )
            else:
                parallelism(
                    function=writeFastas,
                    args=fastasToWrite,
                    chunkSize=1,
                    description="Writing Split Fasta Files...",
                )

        else:
            raise TypeError(
                f"Type {fileSuffix} is not supported. Only .fasta files are asupported."
            )
    else:
        raise FileNotFoundError(f"Fasta File not found: {inputFasta}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits a fasta file containing multiple sequences into separate files."
    )
    parser.add_argument(
        "INPUTFASTA", type=pathlib.Path, help="The fasta file to split."
    )
    parser.add_argument(
        "-o",
        "--OUTPUTDIR",
        type=pathlib.Path,
        default=None,
        help="The directory to output split fasta files to. Defaults to ./splitFastas.",
    )
    parser.add_argument(
        "-O",
        "--OVERWRITE",
        action="store_true",
        help="If present, allows already existing output files at the same path to be overwritten.",
    )
    args = parser.parse_args()

    main(
        inputFasta=args.INPUTFASTA,
        outputDir=args.OUTPUTDIR,
        overwrite=args.OVERWRITE,
    )
