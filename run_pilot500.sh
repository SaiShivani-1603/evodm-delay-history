mkdir -p logs
for c in state_0 state_2 history_2; do
  m=${c%_*}; d=${c#*_}
  for s in 0 1 2 3; do
    python run_baseline.py $s 500 1 $d $m > logs/${m}_seed${s}_ep500_d${d}.log 2>&1 &
  done
  wait
done
echo ALL DONE
