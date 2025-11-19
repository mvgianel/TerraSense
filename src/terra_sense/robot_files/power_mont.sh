#!/usr/bin/env bash
out=${1:-hwmon2_power.csv}
echo "ts_s,voltage_mV,current_mA,power_uW" > "$out"
while :; do
  ts=$(date +%s.%3N)
  v=$(cat /sys/class/hwmon/hwmon2/in1_input)
  c=$(cat /sys/class/hwmon/hwmon2/curr1_input)
  p=$(cat /sys/class/hwmon/hwmon2/power1_input)
  echo "$ts,$v,$c,$p" >> "$out"
  sleep 0.1
done
