# Local Deployment

## Paths
- Project: `~/neurotrom-crypto`
- Data: `~/Library/Application Support/NeuroTrom-Crypto/`
- Logs: `~/Library/Application Support/NeuroTrom-Crypto/logs/`
- LaunchAgents: `~/Library/LaunchAgents/com.neurotrom.crypto.*`

## Ports
- API: `127.0.0.1:8775` (must not conflict with FinVest's 8765 or 8790)

## LaunchAgents
- `com.neurotrom.crypto.api` — API + dashboard
- `com.neurotrom.crypto.collector` — WebSocket live collector
- `com.neurotrom.crypto.daily` — daily pipeline at 20:00 IST

## Install
```bash
cp launchd/com.neurotrom.crypto.*.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.neurotrom.crypto.api.plist
launchctl load ~/Library/LaunchAgents/com.neurotrom.crypto.collector.plist
launchctl load ~/Library/LaunchAgents/com.neurotrom.crypto.daily.plist
```

## Uninstall
```bash
launchctl unload ~/Library/LaunchAgents/com.neurotrom.crypto.api.plist
launchctl unload ~/Library/LaunchAgents/com.neurotrom.crypto.collector.plist
launchctl unload ~/Library/LaunchAgents/com.neurotrom.crypto.daily.plist
rm ~/Library/LaunchAgents/com.neurotrom.crypto.*.plist
```
