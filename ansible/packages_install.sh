# Install apt packages on LAN hosts through Slade's apt cache (no internet needed on the hosts).
# Usage: ./packages_install.sh <host|group> <package> [package...]    e.g. ./packages_install.sh rosey htop iperf3
if [ $# -lt 2 ]; then echo "Usage: $0 <host|group> <package> [package...]"; exit 1; fi
target=$1; shift
ansible-playbook -i mlss_inventory.yaml packages_install.yaml -e "target=$target" -e "packages='$*'"
