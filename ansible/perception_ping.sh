echo "[INFO] This script also checks passwordless sudo access from ansible."
echo "		SUCCESS = connectivity + passwordless sudo are available"
ansible all -i mlss_inventory.yaml -m ping --become
