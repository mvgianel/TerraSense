#!/usr/bin/env bash
set -e
FOLDER=~/rosbaggies   # contains subdirs like "sand_01/", "grass_02/"
TOPIC=/terrain_class

for cls in cobblestonebrick dirtground grass pavement sand stairs; do
  for bagdir in "$FOLDER"/${cls}*/; do
    [ -d "$bagdir" ] || continue
    echo "▶ Playing $bagdir (true class: $cls)…"

    # Find the first .db3 file in the directory
    bagfile=$(find "$bagdir" -maxdepth 1 -type f -name "${cls}*.db3" | head -1)

    # record predictions
    outdir="$FOLDER/preds"
    outfile="$outdir/$(basename "$bagfile" .db3)_preds.txt"
    ros2 topic echo $TOPIC --no-arr >"$outfile" &
    PRED_PID=$!

    # play the bag (once)
    ros2 bag play "$bagfile"

    # stop recording
    kill $PRED_PID
  done
done
