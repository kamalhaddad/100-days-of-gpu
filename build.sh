#!/bin/bash
day=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --day)
            day="$2"
            shift 2 # Move past argument and its value
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$day" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: $0 --day <day>"
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "$(readlink -f -- "$0")")" && pwd)"

cd "$SCRIPT_DIR"
rm -rf ./build
mkdir -p build
cd build
cmake ..
make -j${nprocs}
TARGET_EXEC="day${day}/day${day}"

"$TARGET_EXEC"
