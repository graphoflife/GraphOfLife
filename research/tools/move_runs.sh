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
#
# Nothing is moved while the lab runs: the script holds the lab's own lock for
# as long as it moves, so the lab cannot start, or make again, a run whose
# folder is half way to a link. A run someone is advancing all the same (its
# own lock held, say from the server's Start) is left where it is. Once moved, where each run
# stands is noted in the lab's own folder (.lab/elsewhere.json), so that a finished run counts as
# done while its drive is unplugged (2026-10-29).
set -u
cd "/home/stefan/Documents/CodingProjects/GraphOfLife/GraphOfLifeRuns" || exit 1
DEST=$1
shift
mkdir -p "$DEST" .lab || exit 1
exec 9>".lab/lab.lock"
if ! flock -n 9; then echo "the lab is running: pause it, then move"; exit 1; fi
for run in "$@"; do
    if [ -L "$run" ]; then echo "$run: already a link"; continue; fi
    if [ ! -d "$run" ]; then echo "$run: no such run"; continue; fi
    if [ -e "$run/.lock" ] && ! flock -n "$run/.lock" true; then echo "$run: being played, left in place"; continue; fi
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
# Where each moved run stands, noted on this disk while the other is still there to read: a
# finished run then counts as done whether or not the drive is plugged in (gol_lab.remember_moved).
python3 -c 'import sys; sys.path.insert(0, ".."); import gol_lab; print(len(gol_lab.remember_moved(sys.argv[1:])), "moved runs noted")' "$@"
echo "move done"
