#/bin/bash
set -e
test -n "$1"

key_board_id="$1"
outdir="runs/key${key_board_id}"
board="example_boards/key/${key_board_id}.csv"

. rt/bin/activate

mkdir -p "$outdir"

python -m virtual.mwagent -D5 -Sfullsatsolver "$board"
python -m analyze_log "$outdir/fullsat.txt"

python -m virtual.mwagent -D5 -Smcdfssolver "$board"
python -m analyze_log "$outdir/mcdfs.txt"

python -m virtual.mwagent -D5 -Smcdfssolver "$board" max_vars=100
python -m analyze_log "$outdir/mcdfs_v100.txt"

python -m virtual.mwagent -D5 -Smcdfssolver "$board" max_vars=500
python -m analyze_log "$outdir/mcdfs_v500.txt"

python -m virtual.mwagent -D5 -Smcdfssolver "$board" max_vars=10000
python -m analyze_log "$outdir/mcdfs_nomc.txt"

python -m virtual.mwagent -D5 -Smcsatsolver "$board"
python -m analyze_log "$outdir/mcsat.txt"

python -m virtual.mwagent -D5 -Smcsatsolver "$board" max_vars=100
python -m analyze_log "$outdir/mcsat_v100.txt"

python -m virtual.mwagent -D5 -Smcsatsolver "$board" max_vars=500
python -m analyze_log "$outdir/mcsat_v500.txt"

# mcsat_nomc should be equivalent to fullsat; skipped

python -m describe_infer_times "runs/stats_key${key_board_id}.txt" "$outdir" \
	fullsat mcdfs mcdfs_v100 mcdfs_v500 mcdfs_nomc mcsat mcsat_v100 mcsat_v500
