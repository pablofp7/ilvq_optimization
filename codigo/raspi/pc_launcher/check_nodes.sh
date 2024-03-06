#!/bin/bash

for i in {0..N-1}
do
  hostname="nodo${i}.local"
  if ping -c 1 -W 1 "${hostname}" &>/dev/null
  then
    echo "${hostname} is online"
  else
    echo "${hostname} is offline"
  fi
done
