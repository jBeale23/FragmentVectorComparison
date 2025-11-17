#include <asm-generic/errno-base.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <glob.h>
#include <libgen.h>
#include <limits.h>
#include <linux/limits.h>
#include <math.h>
#include <omp.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#ifdef __linux__
#include <unistd.h>
#endif

/* HACK: Manually set PATH_MAX to 4096 if a given POSIX system doesn't define it
 * in <limits.h>
 */

/* WARNING: This hack does not work on Windows, which imposes a strict path
 * maximum of 260 characters. Also, please just don't use this code on Windows
 * without a rewrite, even if it somehow compiles without errors, it's optimized
 * for a POSIX system and uses a large number of POSIX tricks and libraries that
 * Windows just doesn't have, and will likely behave in unexpected ways or just
 * SEGFAULT repeatedly.
 */
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* Usage function called when --help or -h is
 * passed or an unrecognized argument is provided
 */
void usage(char *programName, int exitCode);

// A very small value used for checking floating point inequalities
#define EPSILON 1e-10

/* The number of thermodescriptors expected
 * to be in a properly formatted eScapefile.
 */
#define ESCAPECOLUMNS 8

/* Stores information on a protein vector to make passing it between functions
 * easier. The name element is a pointer to a stripped version of the filepath
 * that the data was read from that must be cleaned up before exiting. It is
 * used so that all structs can be moved in a dynamically allocated array while
 * still maintining identifiable information for making the final output files.
 * The size element is determined while loading the eScape prediction and is
 * used to avoid needing to recalculate vector size during later indexing. The
 * remaining eScapePrediction element holds a pointer to the data array loaded
 * from file and must be cleaned up before exiting.
 */
typedef struct {
  double *eScapePrediction;
  size_t size;
  char *name;
} proteinVector;

// Helper function to automate freeing allocated memory for proteinVectors.
void freeProteinVector(proteinVector *vec);

/* Stores information on a region where the cosine similarity of two protein
 * vectors was above the specified significance threshold. Because all of the
 * information it holds is allocated on the stack or managed in a proteinVector
 * struct, only the pointer to the matchRegion struct itself must be freed.
 */
typedef struct {
  long double cosineSimilarity;
  size_t proteinOneIndex;
  size_t proteinTwoIndex;
  char *proteinOneName;
  char *proteinTwoName;
} matchRegion;

/* Helper function for swapping the matchRegions referenced by two pointers.
 * Used in heapSort and the inline quickSort implementation in introSort.
 */
static inline void swap(matchRegion *a, matchRegion *b);

/* Used to determine the pivot point for quickSort in a manner that achieves an
 * upper bound of O(nlog(n)) time complexity while avoiding the overhead of
 * pseudorandom number generation. Importantly it also performs in place sorting
 * of a, b, and their midpoint while it does so. See Wikipedia for an
 * explanation of why this method is superior to pivoting on the first or last
 * digit: https://en.wikipedia.org/wiki/Quicksort
 */
static inline size_t medianThree(matchRegion array[], size_t a, size_t b);

// Helper function that performs recursive heaping for heapSort
void heapify(matchRegion array[], size_t arrayLength, size_t index);

/* Unstably sorts an array in place by recursively finding the largest number
 * within the array via heaping and moving it to the end of the array. This is
 * used as a fallback by the introsort implementation below when the recursion
 * depth of quickSort has exceeded 2log(n). See Wikipedia for an explanation of
 * the algorithm: https://en.wikipedia.org/wiki/Heapsort
 */
static inline void heapSort(matchRegion array[], size_t arrayLength);

/* Stably sorts an array in place by sliding through the array, moving all
 * values larger than the currently selected value to its right. This is used as
 * a fallback by the introsort implementation below when the length of an array
 * to be sorted is less than 16, as its performance below that length is
 * generally better than quickSort or heapSort. See Wikipedia for an explanation
 * of the algorithm. https://en.wikipedia.org/wiki/Insertion_sort
 */
static inline void insertionSort(matchRegion array[], size_t arrayLength);

// Helper function that performs reursive sorting for introSort
void introSortUtil(matchRegion array[], size_t arrayLength, size_t maxDepth);

/* Unstably sorts an array in place with O(nlog(n)) time complexity. An inline
 * implementation of quickSort is used as the default sorting method, falling
 * back to heapSort after 2log(n) recursions of quickSort and to insertionSort
 * when the length of a subarray is less than 16. See Wikipedia for an
 * explanation of the algorithm: https://en.wikipedia.org/wiki/Introsort
 */
static inline void introSort(matchRegion array[], size_t arrayLength);

/* Performs a sliding window calculation of cosine similarity between every
 * possible window of two proteins. The resulting matches are then sorted, and
 * those at or above the specified percentile of matches are then written to
 * file. The output format follows the convention of
 * /path/to/outputDir/proteinOneName/proteinTwoName.tsv
 */
static inline int slidingWindowCosineSimilarity(proteinVector *proteinVectorOne,
                                                proteinVector *proteinVector,
                                                size_t windowSize,
                                                double percentile,
                                                char *outputDir);

// Prints a matchRegion struct to file
int printMatchRegions(matchRegion *matches, size_t matchesArraySize,
                      size_t windowSize, double percentile,
                      char *outputFilePath);

// Internal directory creation helper for recursive directory making
int mkdirInternal(const char *path, mode_t mode);

// Recursively makes directories in a path with the specified permission
int mkdirRecursive(const char *path, mode_t mode);

// Manually normalizes .. and . in paths, even if the path doesn't exist
char *normalizePath(const char *path);

/* Joins provided file paths, canonicalizing them if the joined path doesn't
 * contain a globbing pattern, even if the joined path doesn't exist.
 */
char *joinPath(size_t count, ...);

// Globs files in the provided path that match the provided glob pattern.
int globFiles(glob_t *globResult, char *inputDirPath, char *globPattern);

// Returns the basename of a provided filepath without extensions
static inline char *stripFilename(char *path);

/* Determines how many lines are in a file by counting the number of
 * newline characters present. If the last character of the file isn't a
 * newline, +1 is added to the count to ensure the last line is not missed.
 */
static inline size_t fileLineCount(FILE *file);

// Packs binary data into unsigned characters
static inline void packBinaryData(unsigned char *data, size_t row,
                                  size_t column, size_t bytesPerRow);

// Unpacks binary data from unsigned characters as integers
static inline int unpackBinaryData(const unsigned char *data, size_t row,
                                   size_t column, size_t bytesPerRow);

/* Reads an eScape prediction file into the proteinVector struct defined above.
 * For the eScapePrediction array, positions 0-7 correspond to the columns of
 * eScape_predictor_REP.pl Native: dG, dHap, dHp, TdS; Denatured: dG, dHap, dHp,
 * TdS for the first residue, then 8-15 for the second, so on, and so forth.
 */
proteinVector *readProteinVector(char *filePath);

// Flags set by command line arguments
static int overwriteFlag;

int main(int argc, char *argv[]) {
  // Structure definition for command line argument parsing
  struct option longOpts[] = {{"help", no_argument, NULL, 'h'},
                              {"windowSize", required_argument, NULL, 'w'},
                              {"percentile", required_argument, NULL, 'p'},
                              {"output", required_argument, NULL, 'o'},
                              {"overwrite", no_argument, &overwriteFlag, 1},
                              {0, 0, 0, 0}};
  int requiredArguments = 2;
  char *cwd = getcwd(NULL, 0);
  char *outputDir = joinPath(2, cwd, "vectorComparisonOutput");
  if (!outputDir) {
    fprintf(stderr, "ERROR: Failed to construct output directory path\n");
    free(cwd);
    return 1;
  }
  free(cwd);
  double percentile = 0.9999;
  size_t windowSize = 0;
  int c;
  int longOptsIndex = 0;
  while ((c = getopt_long(argc, argv, "ho:p:w:", longOpts, &longOptsIndex)) !=
         -1) {
    switch (c) {
    case 0:
      break;
    case 'h':
      usage(argv[0], 0);
      break;
    case 'o':
      if (optarg == NULL || *optarg == '\0') {
        fputs("ERROR: No output path was provided\n", stderr);
        return 1;
      }
      outputDir = strdup(optarg);
      break;
    case 'p':
      percentile = atof(optarg);
      if (0 > percentile || 1 < percentile) {
        fputs("ERROR: Percentile must be a float between 0 and 1\n", stderr);
        return 1;
      }
      break;
    case 'w':
      if (strcmp(optarg, "length") == 0) {
        /* A value of zero when passed to the cosine similarity
         * comparison is automatically parsed to be the length of the
         * shorter protein. This value is only possible to set as the window
         * size when the user doesn't provide the windowSize flag, or
         * explicitly provides it with a value of 'length'.
         */
        windowSize = 0;
        break;
      }
      windowSize = (size_t)atoi(optarg);
      if (windowSize <= 0) {
        fputs(
            "ERROR: Window Size must be an integer greater than 0 or 'length'",
            stderr);
        return 1;
      }
      break;
    case '?':
      usage(argv[0], 1);
      break;
    default:
      usage(argv[0], 0);
      break;
    }
  }
  if (argc - optind == requiredArguments) {
    /* NOTE: Checks if a limited number of CPUS has been allocated by SLURM, and
     * if it has, limits the maximum number of threads to that limited number.
     */
    char *slurmEnv = getenv("SLURM_CPUS_PER_TASK");
    if (slurmEnv) {
      int cpuLimit = atoi(slurmEnv);
      if (cpuLimit > 0)
        omp_set_num_threads(cpuLimit);
    }
    if (mkdirRecursive(outputDir, 0777) != 0) {
      fprintf(stderr, "ERROR: Failed to create output directory %s\n",
              outputDir);
      free(outputDir);
      return 1;
    }
    glob_t predictionFilesGroupOne;
    glob_t predictionFilesGroupTwo;
    int ret =
        globFiles(&predictionFilesGroupOne, argv[optind], "*.eScapev8_pred*");
    if (ret != 0) {
      fprintf(stderr, "ERROR: Failed to glob eScape prediction files from %s\n",
              argv[optind]);
      return 1;
    }
    ret = globFiles(&predictionFilesGroupTwo, argv[optind + 1],
                    "*.eScapev8_pred*");
    if (ret != 0) {
      fprintf(stderr, "ERROR: Failed to glob eScape prediction files from %s\n",
              argv[optind + 1]);
      return 1;
    }
    proteinVector **proteinVectorsGroupOne =
        malloc(predictionFilesGroupOne.gl_pathc * sizeof(proteinVector));
    if (!proteinVectorsGroupOne) {
      globfree(&predictionFilesGroupOne);
      fprintf(stderr,
              "ERROR: Failed to allocate memory for prediction files in "
              "directory %s\n",
              argv[optind]);
      return ENOMEM;
    }
    int loadFailed = 0;
#pragma omp parallel for
    for (size_t i = 0; i < predictionFilesGroupOne.gl_pathc; i++) {
      proteinVector *vec =
          readProteinVector(predictionFilesGroupOne.gl_pathv[i]);
      if (!vec) {
#pragma omp atomic
        loadFailed |= 1;
      } else {
        proteinVectorsGroupOne[i] = vec;
      }
    }
    if (loadFailed != 0) {
#pragma omp parallel for
      for (size_t i = 0; i < predictionFilesGroupOne.gl_pathc; i++) {
        freeProteinVector(proteinVectorsGroupOne[i]);
      }
      globfree(&predictionFilesGroupOne);
      globfree(&predictionFilesGroupTwo);
      free(proteinVectorsGroupOne);
      fprintf(stderr, "ERROR: Failed to load protein vectors from %s\n",
              argv[optind]);
      return 1;
    }
    int makeOutputFailed = 0;
#pragma omp parallel for
    for (size_t i = 0; i < predictionFilesGroupOne.gl_pathc; i++) {
      char *outputPath =
          joinPath(2, outputDir, proteinVectorsGroupOne[i]->name);
      if (!outputPath) {
        fprintf(stderr, "ERROR: Failed to make output path for protein %s\n",
                proteinVectorsGroupOne[i]->name);
#pragma omp atomic
        makeOutputFailed |= 1;
      }
      if (mkdirRecursive(outputPath, 0777) != 0) {
        fprintf(stderr,
                "ERROR: Failed to create output directory for protein %s\n",
                proteinVectorsGroupOne[i]->name);
#pragma omp atomic
        makeOutputFailed |= 1;
      }
      free(outputPath);
    }
    if (makeOutputFailed != 0) {
#pragma omp parallel for
      for (size_t i = 0; i < predictionFilesGroupOne.gl_pathc; i++) {
        freeProteinVector(proteinVectorsGroupOne[i]);
      }
      free(outputDir);
      globfree(&predictionFilesGroupOne);
      globfree(&predictionFilesGroupTwo);
      free(proteinVectorsGroupOne);
      fprintf(
          stderr,
          "ERROR: Failed to create output directories for proteins from %s\n",
          argv[optind]);
      return 1;
    }
    proteinVector **proteinVectorsGroupTwo =
        malloc(predictionFilesGroupTwo.gl_pathc * sizeof(proteinVector));
    if (!proteinVectorsGroupTwo) {
#pragma omp parallel for
      for (size_t i = 0; i < predictionFilesGroupOne.gl_pathc; i++) {
        freeProteinVector(proteinVectorsGroupOne[i]);
      }
      free(outputDir);
      globfree(&predictionFilesGroupOne);
      globfree(&predictionFilesGroupTwo);
      free(proteinVectorsGroupOne);
      fprintf(stderr,
              "ERROR: Failed to allocate memory for prediction files in "
              "directory %s\n",
              argv[optind]);
      return ENOMEM;
    }
    loadFailed = 0;
#pragma omp parallel for
    for (size_t i = 0; i < predictionFilesGroupTwo.gl_pathc; i++) {
      proteinVector *vec =
          readProteinVector(predictionFilesGroupTwo.gl_pathv[i]);
      if (!vec) {
#pragma omp atomic write
        loadFailed = 1;
      } else {
        proteinVectorsGroupTwo[i] = vec;
      }
    }
    if (loadFailed != 0) {
#pragma omp parallel for
      for (size_t i = 0; i < predictionFilesGroupOne.gl_pathc; i++) {
        freeProteinVector(proteinVectorsGroupOne[i]);
      }
#pragma omp parallel for
      for (size_t i = 0; i < predictionFilesGroupTwo.gl_pathc; i++) {
        freeProteinVector(proteinVectorsGroupTwo[i]);
      }
      free(outputDir);
      globfree(&predictionFilesGroupOne);
      globfree(&predictionFilesGroupTwo);
      free(proteinVectorsGroupOne);
      free(proteinVectorsGroupTwo);
      fprintf(stderr, "ERROR: Failed to load protein vectors from %s\n",
              argv[optind + 1]);
      return 1;
    }
    for (size_t i = 0; i < predictionFilesGroupOne.gl_pathc; i++) {
#pragma omp parallel for
      for (size_t j = 0; j < predictionFilesGroupTwo.gl_pathc; j++) {
        slidingWindowCosineSimilarity(proteinVectorsGroupOne[i],
                                      proteinVectorsGroupTwo[j], windowSize,
                                      percentile, outputDir);
      }
    }
#pragma omp parallel for
    for (size_t i = 0; i < predictionFilesGroupOne.gl_pathc; i++) {
      freeProteinVector(proteinVectorsGroupOne[i]);
    }
#pragma omp parallel for
    for (size_t i = 0; i < predictionFilesGroupTwo.gl_pathc; i++) {
      freeProteinVector(proteinVectorsGroupTwo[i]);
    }
    free(outputDir);
    globfree(&predictionFilesGroupOne);
    globfree(&predictionFilesGroupTwo);
    free(proteinVectorsGroupOne);
    free(proteinVectorsGroupTwo);
  } else if (argc - optind > requiredArguments) {
    fprintf(stderr, "Too many arguments provided.\n");
    free(outputDir);
    usage(argv[0], 1);
  } else {
    fprintf(stderr, "Too few arguments provided.\n");
    free(outputDir);
    usage(argv[0], 1);
  }
  return 0;
}

void usage(char *programName, int exitCode) {
  FILE *out = (exitCode == 0) ? stdout : stderr;
  fputs("Determines the regions of highest energetic similarity between two "
        "groups of protein vectors.\n",
        out);
  fprintf(out,
          "Usage %s: [-h, --help] [-o, --output <file path>] [-p, "
          "--percentile <float>] [-w, --windowSize <int|length>] [--overwrite] "
          "[GROUPONEPREDICTIONDIR] [GROUPTWOPREDICTIONDIR]\n",
          programName);
  fputs("Positional Arguments:\n", out);
  fputs("GROUPONEPREDICTIONDIR\tA directory containing one or more protein "
        "vectors in '.eScapev8_pred' format.\n",
        out);
  fputs("GROUPTWOPREDICTIONDIR\tA directory containing one or more protein "
        "vectors in '.eScapev8_pred' format.\n",
        out);
  fputs("Options:\n", out);
  fputs("  -h, --help\t\tPrint this message and exit.\n", out);
  fputs("  -o, --output\t\tSets the directory in which to output found "
        "matches. Defaults to ./vectorComparisonOutput\n",
        out);
  fputs("  -p, --percentile\tSets the percentile above which regions "
        "should be returned. Can be any float between 0 and 1, inclusive. "
        "Defaults to 99.99th percentile.\n",
        out);
  fputs("  -w, --windowSize\tSets the window size to use for the sliding "
        "window comparison. Can be any integer greater than 0 or 'length'. If "
        "unspecified or 'length', uses the length of the shorter protein.\n",
        out);
  fputs("  --overwrite\t\tIf present, existing output files will be "
        "overwritten.\n",
        out);
  exit(exitCode);
}

static inline void swap(matchRegion *a, matchRegion *b) {
  matchRegion tmp = *a;
  *a = *b;
  *b = tmp;
  return;
}

static inline size_t medianThree(matchRegion array[], size_t a, size_t b) {
  size_t mid = (a + b) / 2;
  if (array[b].cosineSimilarity < array[a].cosineSimilarity)
    swap(&array[a], &array[b]);
  if (array[mid].cosineSimilarity < array[a].cosineSimilarity)
    swap(&array[a], &array[mid]);
  if (array[b].cosineSimilarity < array[mid].cosineSimilarity)
    swap(&array[b], &array[mid]);
  return mid;
}

void heapify(matchRegion array[], size_t arrayLength, size_t index) {
  size_t largest = index;
  size_t left = 2 * index + 1;
  size_t right = 2 * index + 2;

  if (left < arrayLength &&
      array[left].cosineSimilarity > array[largest].cosineSimilarity) {
    largest = left;
  }

  if (right < arrayLength &&
      array[right].cosineSimilarity > array[largest].cosineSimilarity) {
    largest = right;
  }

  if (largest != index) {
    swap(&array[index], &array[largest]);
    heapify(array, arrayLength, largest);
  }
}

static inline void heapSort(matchRegion array[], size_t arrayLength) {
  for (size_t i = arrayLength / 2; i-- > 0;) {
    heapify(array, arrayLength, i);
  }

  for (size_t i = arrayLength - 1; i > 0; i--) {
    swap(&array[0], &array[i]);
    heapify(array, i, 0);
  }
}

static inline void insertionSort(matchRegion array[], size_t arrayLength) {
  /* Start from the second element so that there
   * are elements to the left of the key.
   */
  for (size_t i = 1; i < arrayLength; i++) {
    matchRegion key = array[i];
    size_t j = i;
    /* Move elements of the array[0...i-1] that are greater than the
     * key one position to the right of their current position.
     */
    while (j > 0 && array[j - 1].cosineSimilarity > key.cosineSimilarity) {
      array[j] = array[j - 1];
      j--;
    }
    // Move the key to its correct position.
    array[j] = key;
  }
}

void introSortUtil(matchRegion array[], size_t arrayLength, size_t maxDepth) {
  while (1) {
    if (arrayLength < 16) {
      insertionSort(array, arrayLength);
      return;
    } else if (maxDepth == 0) {
      heapSort(array, arrayLength);
      return;
    } else {
      size_t pivotIndex = medianThree(array, 0, arrayLength - 1);
      swap(&array[pivotIndex], &array[arrayLength - 1]);
      matchRegion pivotValue = array[arrayLength - 1];
      size_t i = 0, j = arrayLength - 2;

      while (i <= j) {
        while (i <= j &&
               array[i].cosineSimilarity < pivotValue.cosineSimilarity)
          i++;
        while (i <= j &&
               array[j].cosineSimilarity > pivotValue.cosineSimilarity)
          j--;
        if (i <= j) {
          swap(&array[i], &array[j]);
          i++;
          if (j > 0)
            j--;
        }
      }

      swap(&array[i], &array[arrayLength - 1]);

      size_t leftSize = i;
      size_t rightSize = arrayLength - i - 1;

      if (leftSize < rightSize) {
        introSortUtil(array, leftSize, maxDepth - 1);
        array = array + i + 1;
        arrayLength = rightSize;
      } else {
        introSortUtil(array + i + 1, rightSize, maxDepth - 1);
        arrayLength = leftSize;
      }

      maxDepth--;
    }
  }
}

static inline void introSort(matchRegion array[], size_t arrayLength) {
  if (arrayLength < 2) {
    return;
  }
  size_t maxDepth = (size_t)(2 * floor(log2((double)arrayLength)));
  introSortUtil(array, arrayLength, maxDepth);
}

static inline int slidingWindowCosineSimilarity(proteinVector *proteinVectorOne,
                                                proteinVector *proteinVectorTwo,
                                                size_t windowSize,
                                                double percentile,
                                                char *outputDir) {
  if (!windowSize) {
    windowSize = (proteinVectorOne->size > proteinVectorTwo->size)
                     ? proteinVectorTwo->size
                     : proteinVectorOne->size;
  }
  size_t outputFileNameLen =
      strlen(proteinVectorTwo->name) + strlen(".tsv") + 1;
  char *outputFileName = malloc(outputFileNameLen);
  if (!outputFileName) {
    fprintf(stderr, "ERROR: Failed to allocate memory for output file name\n");
    return ENOMEM;
  }
  snprintf(outputFileName, outputFileNameLen, "%s.tsv", proteinVectorTwo->name);
  char *outputFilePath =
      joinPath(3, outputDir, proteinVectorOne->name, outputFileName);
  if (!outputFilePath) {
    free(outputFileName);
    fprintf(
        stderr,
        "ERROR: Failed to construct output file path for proteins %s and %s\n",
        proteinVectorOne->name, proteinVectorTwo->name);
    return EINVAL;
  }
  free(outputFileName);
  if (access(outputFilePath, F_OK) == 0 && overwriteFlag == 0) {
    fprintf(stderr, "WARNING: File %s exists and overwrite is false\n",
            outputFilePath);
    free(outputFilePath);
    return 0;
  }
  size_t eScapeWindow = windowSize * ESCAPECOLUMNS;
  size_t matchesArraySize = (proteinVectorOne->size - windowSize + 1) *
                            (proteinVectorTwo->size - windowSize + 1);
  if (proteinVectorOne->size < windowSize ||
      proteinVectorTwo->size < windowSize) {
    fprintf(stderr,
            "ERROR: Window Size %zu is larger than length of protein %s with "
            "length %zu or length of protein %s with length %zu\n",
            windowSize, proteinVectorOne->name, proteinVectorOne->size,
            proteinVectorTwo->name, proteinVectorTwo->size);
    return EINVAL;
  }
  matchRegion *matches = malloc(matchesArraySize * sizeof(matchRegion));
  if (!matches) {
    fprintf(stderr, "ERROR: Failed to allocate memory for match regions\n");
    return ENOMEM;
  }
  for (size_t vectorOnePosition = 0;
       vectorOnePosition <= proteinVectorOne->size - windowSize;
       vectorOnePosition++) {
    long double proteinOneMagnitude = 0.0;
    for (size_t i = vectorOnePosition * ESCAPECOLUMNS;
         i < vectorOnePosition * ESCAPECOLUMNS + eScapeWindow; i++) {
      proteinOneMagnitude += proteinVectorOne->eScapePrediction[i] *
                             proteinVectorOne->eScapePrediction[i];
    }
    proteinOneMagnitude = sqrtl(proteinOneMagnitude);
    for (size_t vectorTwoPosition = 0;
         vectorTwoPosition <= proteinVectorTwo->size - windowSize;
         vectorTwoPosition++) {

      long double proteinTwoMagnitude = 0.0;
      for (size_t i = vectorTwoPosition * ESCAPECOLUMNS;
           i < vectorTwoPosition * ESCAPECOLUMNS + eScapeWindow; i++) {
        proteinTwoMagnitude += proteinVectorTwo->eScapePrediction[i] *
                               proteinVectorTwo->eScapePrediction[i];
      }

      proteinTwoMagnitude = sqrtl(proteinTwoMagnitude);
      long double dotproduct = 0.0;
      for (size_t i = 0; i < eScapeWindow; i++) {
        dotproduct +=
            proteinVectorOne
                ->eScapePrediction[vectorOnePosition * ESCAPECOLUMNS + i] *
            proteinVectorTwo
                ->eScapePrediction[vectorTwoPosition * ESCAPECOLUMNS + i];
      }
      size_t matchIndex =
          vectorOnePosition * (proteinVectorTwo->size - windowSize + 1) +
          vectorTwoPosition;
      matches[matchIndex].cosineSimilarity =
          ((fabsl(proteinOneMagnitude) < EPSILON ||
            fabsl(proteinTwoMagnitude) < EPSILON)
               ? NAN
               : (dotproduct / (proteinOneMagnitude * proteinTwoMagnitude)));
      if (isnan(matches[matchIndex].cosineSimilarity)) {
        free(matches);
        fprintf(stderr,
                "ERROR: The magnitude of protein vector %s or %s was zero\n",
                proteinVectorOne->name, proteinVectorTwo->name);
        return ERANGE;
      }
      matches[matchIndex].proteinOneName = proteinVectorOne->name;
      matches[matchIndex].proteinTwoName = proteinVectorTwo->name;
      matches[matchIndex].proteinOneIndex = vectorOnePosition;
      matches[matchIndex].proteinTwoIndex = vectorTwoPosition;
    }
  }
  introSort(matches, matchesArraySize);
  printMatchRegions(matches, matchesArraySize, windowSize, percentile,
                    outputFilePath);
  free(outputFilePath);
  free(matches);
  return 0;
}

int printMatchRegions(matchRegion *matches, size_t matchesArraySize,
                      size_t windowSize, double percentile,
                      char *outputFilePath) {
  errno = 0;
  FILE *fptr = fopen(outputFilePath, "w");
  if (!fptr) {
    fprintf(stderr, "ERROR: Unable to open file %s: %s\n", outputFilePath,
            strerror(errno));
    return errno;
  }
  fprintf(fptr,
          "#Window Size: %zu\n#Percentile: %f\n#%s[Index]\t%s[Index]\tCosine "
          "Similarity\n",
          windowSize, percentile, matches->proteinOneName,
          matches->proteinTwoName);
  if (fabsl(matches[0].cosineSimilarity - 1) > EPSILON) {
    for (size_t i = (size_t)(percentile * (double)matchesArraySize);
         i < matchesArraySize; i++) {
      fprintf(fptr, "%zu\t%zu\t%Lf\n", matches[i].proteinOneIndex,
              matches[i].proteinTwoIndex, matches[i].cosineSimilarity);
    }
  } else {
    for (size_t i = 0; i < matchesArraySize; i++) {
      fprintf(fptr, "%zu\t%zu\t%Lf\n", matches[i].proteinOneIndex,
              matches[i].proteinTwoIndex, matches[i].cosineSimilarity);
    }
  }
  fclose(fptr);
  return 0;
}

int mkdirInternal(const char *path, mode_t mode) {
  struct stat st;
  errno = 0;
  if (mkdir(path, mode) == 0) {
    return 0;
  }
  if (errno != EEXIST) {
    return -1;
  }
  if (stat(path, &st) != 0) {
    return -1;
  }
  if (!S_ISDIR(st.st_mode)) {
    errno = ENOTDIR;
    return -1;
  }
  return 0;
}

int mkdirRecursive(const char *path, mode_t mode) {
  char *_path = NULL;
  _path = strdup(path);
  if (!_path) {
    return -1;
  }
  for (char *p = _path + 1; *p; p++) {
    if (*p == '/') {
      *p = '\0';
      if (mkdirInternal(_path, mode) != 0) {
        free(_path);
        return -1;
      }
      *p = '/';
    }
  }
  if (mkdirInternal(_path, mode) != 0) {
    free(_path);
    return -1;
  }
  return 0;
}

char *normalizePath(const char *path) {
  if (!path)
    return NULL;

  char *normalized = malloc(PATH_MAX);
  if (!normalized) {
    return NULL;
  }

  if (realpath(path, normalized) == NULL) {
  }

  return normalized;
}

char *joinPath(size_t count, ...) {
  if (count == 0) {
    fputs("ERROR: No path components provided to joinPath()\n", stderr);
    return NULL;
  }

  size_t totalLen = 1;
  va_list args;
  va_start(args, count);
  for (size_t i = 0; i < count; i++) {
    const char *part = va_arg(args, const char *);
    if (!part || part[0] == '\0') {
      va_end(args);
      fprintf(stderr, "ERROR: Empty or null path component at position %zu\n",
              i);
      return NULL;
    }
    totalLen += strlen(part) + 1;
  }
  va_end(args);

  char *joinedPath = malloc(totalLen);
  if (!joinedPath) {
    fprintf(stderr, "ERROR: Failed to allocate %zu bytes for path\n", totalLen);
    return NULL;
  }

  joinedPath[0] = '\0';
  va_start(args, count);
  for (size_t i = 0; i < count; i++) {
    const char *part = va_arg(args, const char *);
    size_t len = strlen(joinedPath);
    int needsSlash = (len > 0 && joinedPath[len - 1] != '/' && part[0] != '/');

    if (needsSlash) {
      strcat(joinedPath, "/");
    }
    strcat(joinedPath, part);
  }
  va_end(args);

  if (strchr(joinedPath, '*') || strchr(joinedPath, '?') ||
      strchr(joinedPath, '[')) {
    return joinedPath;
  }

  errno = 0;
  char *canonical = realpath(joinedPath, NULL);
  if (canonical) {
    free(joinedPath);
    return canonical;
  }

  if (errno == ENOENT) {
    char *prefix = strdup(joinedPath);
    if (!prefix) {
      perror("strdup failed");
      free(joinedPath);
      return NULL;
    }

    for (char *p = prefix + strlen(prefix) - 1; p > prefix; p--) {
      if (*p == '/') {
        *p = '\0';

        char *realPrefix = realpath(prefix, NULL);
        if (realPrefix) {
          const char *suffix = joinedPath + strlen(prefix);

          while (*suffix == '/')
            suffix++;

          size_t combinedLen = strlen(realPrefix) + strlen(suffix) + 2;
          char *unsanitized = malloc(combinedLen);
          if (!unsanitized) {
            fprintf(stderr, "ERROR: Allocation failed for %zu bytes\n",
                    combinedLen);
            free(realPrefix);
            free(prefix);
            free(joinedPath);
            return NULL;
          }

          snprintf(unsanitized, combinedLen, "%s/%s", realPrefix, suffix);
          char *normalized = normalizePath(unsanitized);
          free(unsanitized);
          free(realPrefix);
          free(prefix);
          free(joinedPath);
          return normalized;
        }
      }
    }

    fprintf(stderr, "ERROR: No component of path '%s' exists\n", joinedPath);
    free(prefix);
    free(joinedPath);
    return NULL;
  }

  fprintf(stderr, "ERROR: realpath() failed on path '%s': [%d] %s\n",
          joinedPath, errno, strerror(errno));
  free(joinedPath);
  return NULL;
}

int globFiles(glob_t *globResult, char *inputDirPath, char *globPattern) {
  char *searchPath = joinPath(2, inputDirPath, globPattern);
  if (!searchPath) {
    fprintf(stderr,
            "ERROR: Failed to concatenate glob pattern %s to inputDirPath %s\n",
            inputDirPath, globPattern);
    return 1;
  }
  memset(globResult, 0, sizeof(glob_t));
  int ret;
  ret = glob(searchPath, 0, NULL, globResult);
  free(searchPath);
  if (ret != 0) {
    globfree(globResult);
    if (ret == GLOB_NOMATCH) {
      fprintf(stderr,
              "ERROR: No matches found in path %s with "
              "glob pattern %s\n",
              inputDirPath, globPattern);
      return GLOB_NOMATCH;
    } else {
      fprintf(stderr,
              "ERROR: An error occurred during globbing "
              "which returned code %d\n",
              ret);
      return ret;
    }
  }
  return 0;
}

static inline char *stripFilename(char *path) {
  char *pathCopy = strdup(path);
  char *filename = basename(pathCopy);
  char *dot = strrchr(filename, '.');
  if (dot) {
    *dot = '\0';
  }
  char *strippedFilename = strdup(filename);
  free(pathCopy);
  return strippedFilename;
}

static inline size_t fileLineCount(FILE *file) {
  size_t lines = 0;
  int c;
  int last = 0;

  while ((c = fgetc(file)) != EOF) {
    if (c == '\n') {
      lines++;
    }
    last = c;
  }

  if (lines == 0 && last == 0) {
    rewind(file);
    return 0;
  }

  if (last != '\n') {
    lines++;
  }

  rewind(file);
  return lines;
}

static inline void packBinaryData(unsigned char *data, size_t row,
                                  size_t column, size_t bytesPerRow) {
  size_t byteIndex = row * bytesPerRow + (column / 8);
  size_t bitOffset = column % 8;
  data[byteIndex] |= (1 << bitOffset);
}

static inline int unpackBinaryData(const unsigned char *data, size_t row,
                                   size_t column, size_t bytesPerRow) {
  size_t byteIndex = row * bytesPerRow + (column / 8);
  size_t bitOffset = column % 8;
  return (data[byteIndex] >> bitOffset) & 1;
}

proteinVector *readProteinVector(char *filePath) {
  FILE *file = fopen(filePath, "r");
  if (!file) {
    fprintf(stderr,
            "ERROR: Failed to open eScape prediction "
            "file %s\n",
            filePath);
    return NULL;
  }
#if defined(__linux__)
  int fd = fileno(file);
  int ret = posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
  if (ret != 0) {
    fprintf(stderr,
            "POSIX_FADV_SEQUENTIAL failed with error "
            "code: %s\n",
            strerror(ret));
  }
  ret = posix_fadvise(fd, 0, 0, POSIX_FADV_WILLNEED);
  if (ret != 0) {
    fprintf(stderr,
            "POSIX_FADV_WILLNEED failed with error "
            "code: %s\n",
            strerror(ret));
  }
#endif
  proteinVector *vec = malloc(sizeof(proteinVector));
  if (!vec) {
    fprintf(stderr,
            "ERROR: Failed to allocate memory for "
            "protein vector of eScape "
            "prediction file %s\n",
            filePath);
    return NULL;
  }
  vec->name = stripFilename(filePath);
  vec->size = fileLineCount(file);
  vec->eScapePrediction = malloc(vec->size * ESCAPECOLUMNS * sizeof(double));
  if (!vec->eScapePrediction) {
    freeProteinVector(vec);
    fclose(file);
    fprintf(stderr,
            "ERROR: Failed to allocate memory for "
            "thermodescriptors in eScape "
            "prediction file %s\n",
            filePath);
    return NULL;
  }
  char *line = NULL;
  size_t lineLength = 0;
  ssize_t read;
  size_t lineNum = 0;
  while ((read = getline(&line, &lineLength, file)) != -1 &&
         lineNum < vec->size) {
    if (line[read - 1] == '\n') {
      line[--read] = '\0';
    }
    char *thermodescriptors[ESCAPECOLUMNS];
    char *saveptr;
    char *entry = strtok_r(line, "\t", &saveptr);
    size_t i = 0;
    while (entry && i < ESCAPECOLUMNS) {
      thermodescriptors[i] = entry;
      entry = strtok_r(NULL, "\t", &saveptr);
      i++;
    }
    if (i != ESCAPECOLUMNS) {
      freeProteinVector(vec);
      free(line);
      fclose(file);
      fprintf(
          stderr,
          "ERROR File %s Line %zu: expected %d thermodescriptors, got %zu\n",
          filePath, lineNum + 1, ESCAPECOLUMNS, i);
      return NULL;
    }
    for (size_t col = 0; col < ESCAPECOLUMNS; col++) {
      char *endptr;
      errno = 0;
      double val = strtod(thermodescriptors[col], &endptr);
      if (errno != 0 || *endptr != '\0') {
        freeProteinVector(vec);
        free(line);
        fclose(file);
        fprintf(stderr,
                "ERROR File %s Line %zu, column %zu: "
                "malformed thermodescriptor "
                "'%s'\n",
                filePath, lineNum + 1, col, thermodescriptors[col]);
        return NULL;
      }
      vec->eScapePrediction[lineNum * ESCAPECOLUMNS + col] = val;
    }
    lineNum++;
  }
  if (lineNum != vec->size) {
    freeProteinVector(vec);
    free(line);
    fclose(file);
    fprintf(stderr, "ERROR: Incorrect line count %zu for file %s.\n", lineNum,
            filePath);
    return NULL;
  }
  free(line);
  fclose(file);
  return vec;
}

void freeProteinVector(proteinVector *vec) {
  free(vec->name);
  free(vec->eScapePrediction);
  free(vec);
}
