# Virtual Machine Attack Demo Commands

This document records the commands used to demonstrate the Kali-to-Ubuntu attack
workflow and trigger the Streamlit visualization dashboard.

The demo is designed for a controlled local lab environment:

- Attacker VM: Kali Linux
- Target VM: Ubuntu
- Host system: Windows + VMware Workstation
- Visualization system: Streamlit IDS dashboard

## 1. Check VM IP Addresses

Run on Kali:

```bash
ip -4 addr
```

Run on Ubuntu:

```bash
ip -4 addr
```

Example IP addresses:

```text
Kali:   192.168.99.141
Ubuntu: 192.168.99.140
```

Replace the IP addresses in the following commands with your actual VM IPs.

## 2. Check VMware Shared Folder

In the current demo setup, the Windows `data/vm_bridge` directory is shared
directly to Kali. Therefore, the signal file path on Kali is:

```bash
/mnt/hgfs/vm_bridge/current_scenario.json
```

Check whether the shared folder is mounted:

```bash
ls /mnt/hgfs
```

If `/mnt/hgfs` does not exist or the shared folder is not visible, run:

```bash
sudo mkdir -p /mnt/hgfs
sudo vmhgfs-fuse .host:/ /mnt/hgfs -o allow_other
ls /mnt/hgfs
```

If the shared folder name is `vm_bridge`, the path is correct.

## 3. Normal Access Demo

Run on Kali:

```bash
curl http://192.168.99.140
```

Trigger normal traffic visualization:

```bash
echo '{"scenario":"normal","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":200}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

Check the signal file:

```bash
cat /mnt/hgfs/vm_bridge/current_scenario.json
```

Expected dashboard behavior:

- Attack traffic is 0 or very low.
- Risk status is normal or low risk.
- No red attack path is shown in the topology.

## 4. Port Scan Demo

Run on Kali:

```bash
nmap -sS -T4 192.168.99.140
```

Trigger PortScan visualization:

```bash
echo '{"scenario":"portscan","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":200}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

Expected dashboard behavior:

- Attack type includes `PortScan`.
- Suspicious or attack traffic appears.
- Alert events and topology status update.

## 5. SSH Brute Force Demo

Run on Kali:

```bash
hydra -l yaocheng -P /usr/share/wordlists/rockyou.txt ssh://192.168.99.140 -t 4
```

If the Ubuntu username is not `yaocheng`, replace it with your actual username.

Trigger SSH-Patator visualization:

```bash
echo '{"scenario":"ssh_patator","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":200}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

Expected dashboard behavior:

- Attack type is shown as `SSH-Patator`.
- High-risk alerts appear.
- Attack source and target host are highlighted.

## 6. Web High-Frequency Access / DoS Demo

Run on Kali:

```bash
ab -n 1000 -c 50 http://192.168.99.140/
```

If `ab` is not installed:

```bash
sudo apt update
sudo apt install apache2-utils -y
```

Trigger DoS visualization:

```bash
echo '{"scenario":"dos","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":200}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

Expected dashboard behavior:

- Attack type includes `DoS`.
- High-risk alert events appear.
- Attack traffic and risk distribution update.

## 7. Mixed Attack Demo

Run several attack actions in sequence:

```bash
nmap -sS -T4 192.168.99.140
```

```bash
hydra -l yaocheng -P /usr/share/wordlists/rockyou.txt ssh://192.168.99.140 -t 4
```

```bash
ab -n 1000 -c 50 http://192.168.99.140/
```

Trigger mixed attack visualization:

```bash
echo '{"scenario":"mixed_attack","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":300}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

Expected dashboard behavior:

- Multiple attack types are shown.
- Attack type proportion chart updates.
- Real-time alert table displays high-risk events.
- Topology shows attack source, attack path, and target state.

## 8. Recommended Demo Order

For a presentation or interview, use this order:

```bash
curl http://192.168.99.140
echo '{"scenario":"normal","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":200}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

```bash
nmap -sS -T4 192.168.99.140
echo '{"scenario":"portscan","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":200}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

```bash
hydra -l yaocheng -P /usr/share/wordlists/rockyou.txt ssh://192.168.99.140 -t 4
echo '{"scenario":"ssh_patator","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":200}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

```bash
ab -n 1000 -c 50 http://192.168.99.140/
echo '{"scenario":"dos","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":200}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

```bash
echo '{"scenario":"mixed_attack","attack_ip":"192.168.99.141","target_ip":"192.168.99.140","rows":300}' > /mnt/hgfs/vm_bridge/current_scenario.json
```

## 9. Alternative Path

If you share the entire project directory instead of sharing `vm_bridge`
directly, the signal file path may look like this:

```bash
/mnt/hgfs/ids_project/data/vm_bridge/current_scenario.json
```

In that case, replace:

```bash
/mnt/hgfs/vm_bridge/current_scenario.json
```

with:

```bash
/mnt/hgfs/ids_project/data/vm_bridge/current_scenario.json
```

in all commands above.

