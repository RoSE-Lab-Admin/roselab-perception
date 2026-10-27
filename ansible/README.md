# MLSS Ansible

Run everything from Slade, in this directory. `ansible.cfg` points at `mlss_inventory.yaml`, so `-i` is optional.

| Script | What it does |
|---|---|
| `./perception_ping.sh` | Connectivity + passwordless sudo check |
| `./perception_up.sh` / `./perception_down.sh` | Start / stop the `perception-*` campaign services |
| `./packages_install.sh <host\|group> <pkg>...` | Install apt packages on hosts with no internet (through Slade) |
| `./zenoh_setup.sh` | Install rmw_zenoh, start a Zenoh router on every ROS 2 host at boot, make Zenoh the default RMW |
| `./chrony_setup.sh` | Install/configure chrony (Slade serves, everyone else follows Slade) and install the health monitor |
| `./chrony_check.sh` | Time sync report for every host; exits non-zero if a reachable host is out of sync |

Every wrapper passes extra arguments to `ansible-playbook`, e.g. `--limit rosey`, `--check --diff` (dry run), `--tags check`.

## Offline package installs

The Pis (`pis` inventory group) have no internet; Slade, the NUC and the gantry PCs do. Slade runs **apt-cacher-ng** on port 3142
(`roles/apt_proxy`). Each Pi gets an apt hook (`roles/apt_client`) that sends apt traffic through Slade when it is reachable and goes
direct otherwise. So plain `apt install` works on the Pis too, with correct architecture and dependencies, and each package is
downloaded once for all of them. Hosts outside `pis` install straight from the internet. Move a host in or out of the group and re-run
any setup playbook to add or remove the hook.

```bash
./packages_install.sh rosey htop iperf3            # group or host name from the inventory
./packages_install.sh "nuc,miniforum" ros-jazzy-foxglove-bridge
# .deb files that aren't in any repo (dependencies still resolve through the proxy):
ansible-playbook packages_install.yaml -e target=mastcam_front -e '{"deb_files": ["debs/vendor_1.0_arm64.deb"]}'
```

- **No internet on Slade either (field):** packages that are already cached still install. Set `apt_proxy_offline_mode: true` in
  `group_vars/all.yaml` and re-run any setup playbook. Apt then stops waiting on upstream timeouts. To pre-load the cache before a
  trip, install the packages on the Pis once while Slade is online.
- **Repeatable config:** add a package to the play for its host or group (or to `packages:` in `host_vars/<host>.yaml`), not ad hoc.

## Zenoh (ROS 2 Jazzy middleware)

- Every host in the `ros2` inventory group runs `zenoh-router.service` (`rmw_zenohd`, enabled at boot, restarts on failure).
  ROS nodes connect to their local router; routers link to each other across the LAN.
- Topology (`zenoh_topology` in `roles/zenoh/defaults/main.yaml`):
  - `mesh` (the default): each router connects to every `ros2` host listed above it in the inventory. Every pair gets exactly one link, and
    rover-to-rover traffic does not depend on Slade.
  - `star`: every router connects only to Slade.
- `RMW_IMPLEMENTATION=rmw_zenoh_cpp` is set in `/etc/profile.d/ros2_rmw.sh` (login shells, SSH, and the `perception-*` services, which
  use `bash --login`) and in `/etc/environment` (desktop sessions).
- Router config: `/etc/zenoh/rmw_zenoh_router.env` sets `ZENOH_CONFIG_OVERRIDE` on top of the rmw_zenoh package defaults.
- Health: `./zenoh_setup.sh --tags check` prints, for each host, which routers it is linked to, and warns about mixed rmw_zenoh versions.
  On a host: `systemctl status zenoh-router`, `journalctl -u zenoh-router -f`.
- **Cutover:** Zenoh and FastDDS nodes can't see each other. Switch every ROS host in one run, and restart running nodes.
- **Roll back to FastDDS:** set `rmw_implementation: rmw_fastrtps_cpp` in `group_vars/all.yaml` and re-run `./zenoh_setup.sh`.
  The routers are then stopped and disabled.

## Time sync (chrony)

- Slade (`lan_server` group) serves `chrony_serve_subnet` from `conf.d/serve-rover.conf`.
- Clients use `/etc/chrony/sources.d/mlss_ntp.sources` (`server 192.168.2.10 iburst prefer minpoll 4 maxpoll 6`). The stock internet
  pools are commented out, so every host follows Slade only. The role also sets `makestep 1 -1` and keeps sync logs in `/var/log/chrony`.
- **Runtime monitoring:** `chrony-health.timer` runs `/usr/local/bin/chrony-health` every 60 s on every host. Failures are logged at
  error level and mark the unit failed:
  ```bash
  journalctl -u chrony-health -p warning     # history of sync problems on this host
  sudo chrony-health                         # one-off check (add --json for machine-readable output)
  ```
- **What is checked** (thresholds in `roles/chrony/defaults/main.yaml`):
  - The host is synced to Slade.
  - |offset| ≤ 2 ms.
  - NTP round trip to Slade ≤ 10 ms.
  - chrony's max error bound ≤ 5 ms.
  - No recent polls were lost (warning only).
  - On Slade, the check lists which clients are polling it.

## Network / firewall

Slade runs WSL2 in mirrored networking mode. LAN hosts must reach these ports on 192.168.2.10:
- TCP 3142 (apt proxy, Pis only)
- TCP 7447 (Zenoh)
- UDP 123 (NTP)

If the Hyper-V firewall blocks a port, allow it from an admin PowerShell on Windows:
```powershell
New-NetFirewallHyperVRule -Name MLSS-WSL-TCP -DisplayName "MLSS apt/zenoh" -Direction Inbound `
  -VMCreatorId '{40E0AC32-46A5-438A-A0B2-2B479E8F2E90}' -Protocol TCP -LocalPorts 3142,7447
```
