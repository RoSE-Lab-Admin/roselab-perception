ansible all -i mlss_inventory.yaml -b -m shell -a "systemctl stop 'perception-*'"
