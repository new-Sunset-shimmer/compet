#!/bin/bash
for seed_val in 909 1990 666 1564 18 1609 1228
do
sed -i "s/seed = .*/seed = $seed_val /g" /disk/leesm/yangcw/Genre_competition/util/config/config.py
python ~/run.py
wait
echo all jobs are done!
done
