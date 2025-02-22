#!/bin/bash
src="test4_resultados_ranges"
dst="test4_resultados"

# Create destination directory if it doesn't exist
mkdir -p "$dst"

for file in "$src"/*; do
  base=$(basename "$file")
  # Remove the substring if it exists
  new=$(echo "$base" | sed 's/_limit500_range72\.5-77\.5//g')
  # Move (rename) the file to the destination directory with the new name
  mv "$file" "$dst/$new"
done
