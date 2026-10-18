#!/bin/sh
# Move lab runs to another disk, leaving a link in each one's place, so every
# path that names a run still reaches it while the disk is mounted.
#
#     sh research/tools/move_runs.sh DEST RUN...
#
# For each run: copy it (rsync -a), check the copy file by file — the same
# files, the same bytes (rsync --checksum, nothing may differ), the same count
# and the same total size — then swap the folder for a link to the copy and
# remove the original. A run whose copy does not check out is left as it was.
# First used 2026-10-16, with the author's agreement, to move the largest old
# runs (Experiments 7, 8 and 10) to /media/stefan/Speicher2/GraphOfLifeRuns;
# on 2026-10-18 Experiments 17 to 19 and the pilot batches R, T and V followed.
# A pilot batch is named by its path under GraphOfLifeRuns, e.g. `.pilots/R`.
set -u
cd "/home/stefan/Documents/CodingProjects/GraphOfLife Claude/GraphOfLifeRuns" || exit 1
DEST=$1
shift
mkdir -p "$DEST" || exit 1
for run in "$@"; do
    if [ -L "$run" ]; then echo "$run: already a link"; continue; fi
    if [ ! -d "$run" ]; then echo "$run: no such run"; continue; fi
    mkdir -p "$DEST/$(dirname "$run")" || { echo "$run: no room made for it there, left in place"; continue; }
    if ! rsync -a "$run/" "$DEST/$run/"; then echo "$run: copy failed, left in place"; continue; fi
    differs=$(rsync -a --checksum --dry-run --itemize-changes "$run/" "$DEST/$run/" | grep -v '^\.d\.\.t')
    files_here=$(find "$run" -type f | wc -l)
    files_there=$(find "$DEST/$run" -type f | wc -l)
    bytes_here=$(du -sb "$run" | cut -f1)
    bytes_there=$(du -sb "$DEST/$run" | cut -f1)
    if [ -n "$differs" ] || [ "$files_here" != "$files_there" ]; then
        echo "$run: copy differs ($files_here files here, $files_there there), left in place"
        continue
    fi
    if mv "$run" "$run.moving" && ln -s "$DEST/$run" "$run"; then
        rm -rf "$run.moving"
        echo "$run: moved ($files_here files, $bytes_here bytes here, $bytes_there there)"
    else
        [ -d "$run.moving" ] && [ ! -e "$run" ] && mv "$run.moving" "$run"
        echo "$run: could not swap, left in place"
    fi
done
echo "move done"
