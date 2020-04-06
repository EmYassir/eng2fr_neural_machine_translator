#!/bin/bash


# Concatenates n1 lines of file1 and n2 lines from file2
concatenate_files()
{
  if [ $# -ne 5 ]; then
    echo "--usage: concatenate_files <file 1> <number of lines in file 1> <file2> <number of lines in file 2> <output>"
    return 1
  fi
  FILE1=$1
  shift
  N1=$1
  shift  
  FILE2=$1
  shift
  N2=$1
  shift
  OUTPUT=$1
  sed -n -e "1,${N1}p" $FILE1 > $OUTPUT
  sed -n -e "1,${N2}p" $FILE2 >> $OUTPUT
}





# Main
concatenate_files $@
