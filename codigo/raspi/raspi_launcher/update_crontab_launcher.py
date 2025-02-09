#!/usr/bin/env python3
import sys
import subprocess

# Ensure a launcher suffix is provided
if len(sys.argv) != 2:
    print("Usage: update_crontab_launcher.py <launcher> (i.e., 'launcherv1, launcherv2, vfdt_launcher, nb_launcher, etc.')")
    sys.exit(1)

launcher = sys.argv[1]

# Define the new crontab line. Adjust paths if necessary.
crontab_line = f"@reboot sleep 120 && cd /home/pablo/ilvq_optimization/codigo/raspi/raspi_launcher && echo  && nohup python3 super_launcher.py {launcher} > super_launcher.log 2>&1 &"
crontab_line = f"@reboot sleep 120 && cd /home/pablo/ilvq_optimization/codigo/raspi && echo \"Starting super_launcher at $(date)\" >
                super_launcher.log 2>&1 && python3 super_launcher.py {launcher} >> super_launcher.log 2>&1"


# Get the current crontab entries
try:
    result = subprocess.run("crontab -l", shell=True, capture_output=True, text=True, check=True)
    current_crontab = result.stdout.splitlines()
except subprocess.CalledProcessError:
    # If no crontab exists, start with an empty list
    current_crontab = []

# Remove any existing crontab lines that mention super_launcher.py
new_crontab = [line for line in current_crontab if "super_launcher.py" not in line]

# Append the new crontab entry
new_crontab.append(crontab_line)

# Prepare the new crontab content
new_crontab_content = "\n".join(new_crontab) + "\n"

# Update the crontab
proc = subprocess.run("crontab -", input=new_crontab_content, text=True, shell=True)
if proc.returncode == 0:
    print("Crontab updated successfully:")
    print(crontab_line)
else:
    print("Failed to update crontab.")
