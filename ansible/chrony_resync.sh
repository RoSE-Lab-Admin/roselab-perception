#!/usr/bin/env bash
# Time sync gate before data collection: runs pi_check.yaml and, while it fails, restarts chrony on Slade and on
# every reachable checked host (chrony_resync.yaml). Exits 0 once every host passes, 1 if they never do.
#   ./chrony_resync.sh
#   ATTEMPTS=10 ./chrony_resync.sh
#   ./chrony_resync.sh --limit slade,wheelcam    # keep slade in any --limit so the server is restarted too
cd "$(dirname "$0")" || exit 1
attempts=${ATTEMPTS:-5}

for ((i = 1; i <= attempts; i++)); do
  if ansible-playbook -i mlss_inventory.yaml pi_check.yaml "$@"; then
    echo "All hosts are synced to Slade."
    exit 0
  fi
  echo ">>> Time sync check failed: restarting chrony (attempt $i of $attempts)"
  ansible-playbook -i mlss_inventory.yaml chrony_resync.yaml "$@"
done

if ansible-playbook -i mlss_inventory.yaml pi_check.yaml "$@"; then
  echo "All hosts are synced to Slade."
  exit 0
fi
echo ">>> Still out of sync after $attempts chrony restarts. Do not start data collection; see the report above." >&2
exit 1
