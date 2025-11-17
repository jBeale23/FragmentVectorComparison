#!/usr/bin/env bash

function get_script_dir() {
    local SOURCE_PATH="${BASH_SOURCE[0]}"
    local SYMLINK_DIR
    local SCRIPT_DIR
    # Resolve symlinks recursively
    while [ -L "$SOURCE_PATH" ]; do
        # Get symlink directory
        SYMLINK_DIR="$( cd -P "$( dirname "$SOURCE_PATH" )" >/dev/null 2>&1 && pwd )"
        # Resolve symlink target (relative or absolute)
        SOURCE_PATH="$(readlink "$SOURCE_PATH")"
        # Check if candidate path is relative or absolute
        if [[ $SOURCE_PATH != /* ]]; then
            # Candidate path is relative, resolve to full path
            SOURCE_PATH=$SYMLINK_DIR/$SOURCE_PATH
        fi
    done
    # Get final script directory path from fully resolved source path
    SCRIPT_DIR="$(cd -P "$( dirname "$SOURCE_PATH" )" >/dev/null 2>&1 && pwd)"
    echo "$SCRIPT_DIR"
}

function noArgs() {
	if ! [ -z "$OPTARG" ]; then
		echo "option requires no arguments -- $OPT"
		exit 1
	fi
}

function usage() {
	echo "Usage: findBestMatches.sh [-h, --help]"
	echo "Automatically sets up required conda environment."
  	echo "Options:"
  	echo "  -h, --help	Show this help message and exit."
}

while getopts "h-:" OPT; do
	if [ "$OPT" = "-" ]; then
		OPT="${OPTARG%%=*}"
		OPTARG="${OPTARG#"$OPT"}"
		OPTARG="${OPTARG#=}"
	fi
	case $OPT in
		h | help)
			noArgs
			usage
			exit 0
			;;
		\?)
			usage
     		exit 1
			;;
		*)
			echo "illegal option -- $OPT" >&2
			usage
			exit 1
			;;
	esac
done

# Ensures conda (of some form) is installed
if ! which conda ; then
	echo "Conda was not found, please install Conda. See the following link for details:\nhttps://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html" >&2
	exit 1
fi

# Finds where the script directory is for aliasing
scriptDir=$(get_script_dir)

# Ensures the required conda environment exists 
if ! conda env list | grep -q '^FVC\b' ; then
	conda env create -y --file "${scriptDir}/reqs.yaml"
else
	conda env update --prune --file "${scriptDir}/reqs.yaml"
fi

# Builds cFVC using the provided makefile
cd ${scriptDir}/src && conda run -n FVC make && cd ${scriptdir}

echo "Setup complete."
