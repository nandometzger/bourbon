#!/bin/bash
PYTHON=/scratch3/metzgern/random/Popcorn/PopMapEnv/bin/python
SCRIPT=predict_timeseries.py

# Format: "lat | lon | size | display_name"
declare -a locs=(
    "0.317039 | 32.599400 | 20000 | Kampala, Uganda"
    "-1.947047 | 30.074058 | 20000 | Kigali, Rwanda"
    "-2.503655 | 28.855376 | 20000 | Bukavu, DRC"
    "-3.356942 | 29.364356 | 10000 | Bujumbura, Burundi"
    "-2.877806 | 32.230078 | 10000 | Geita, Tanzania"
    "3.027432 | 30.910985 | 10000 | Arua, Uganda"
    "-10.720634 | 25.501585 | 20000 | Kolwezi, DRC"
    "-2.517776 | 32.901805 | 10000 | Mwanza, Tanzania"
    "-6.174947 | 35.743566 | 10000 | Dodoma, Tanzania"
    "-1.598689 | 29.172785 | 3000 | Goma Refugee Camp 1, DRC"
    "-1.617167 | 29.237954 | 3000 | Goma Refugee Camp 2, DRC"
    "-1.500870 | 29.628970 | 10000 | Musanze, Rwanda"
    "-0.778259 | 30.947134 | 5000 | Nakivale, Uganda"
    "-0.934564 | 30.751842 | 5000 | Oruchinga, Uganda"
    "1.557291 | 30.241213 | 10000 | Bunia, DRC"
    "-3.427823 | 29.924803 | 10000 | Gitega, Burundi"
    "1.433063 | 31.350127 | 10000 | Hoima, Uganda"
)

mkdir -p showcases

for loc in "${locs[@]}"
do
   # Split by |
   IFS='|' read -r lat lon size name <<< "$loc"
   
   # Trim whitespace
   lat=$(echo $lat | xargs)
   lon=$(echo $lon | xargs)
   size=$(echo $size | xargs)
   name=$(echo $name | xargs)

   # Create sanitized folder name
   folder_name=$(echo $name | tr -cd '[:alnum:] ' | tr ' ' '_')

   echo "🥃 Processing $name ($folder_name): ($lat, $lon) @ ${size}m"
   $PYTHON $SCRIPT --lat $lat --lon $lon --size_meters $size --ensemble 5 --provider mpc \
          --out_dir "showcases_v2/$folder_name" --vmax 2.0 --name "$name"
done
