for x in {6..0}; do
  sudo ./enable_cores.sh
  sudo ./disable_cores.sh $x
  ./run.sh
done
