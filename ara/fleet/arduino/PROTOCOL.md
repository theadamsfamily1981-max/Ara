# ara:// Serial Protocol for Arduino Nodes

Simple JSON-over-serial protocol for Arduino ↔ Ara communication.

## Physical Layer

- **Baud rate**: 115200 (default)
- **Format**: 8N1
- **Line ending**: `\n` (newline)
- **Message format**: Single-line JSON

## Message Types

### From Ara → Arduino

#### Heartbeat (Watchdog)
```json
{"cmd":"hb"}
```
Arduino must receive this within timeout or trigger watchdog action.

#### Arm/Disarm Relay
```json
{"cmd":"arm","relay":1}
{"cmd":"disarm","relay":1}
```

#### Trigger Relay (Power Cycle)
```json
{"cmd":"cycle","relay":1,"off_ms":5000}
```
Turn off relay 1, wait 5 seconds, turn back on.

#### Kill Power (Emergency)
```json
{"cmd":"kill","relay":1}
```
Turn off and stay off until explicit `arm`.

#### Set LED State
```json
{"cmd":"led","r":255,"g":0,"b":0}
{"cmd":"led","pattern":"pulse","color":"blue"}
```

#### Query Status
```json
{"cmd":"status"}
```

### From Arduino → Ara

#### Sensor Reading
```json
{"type":"sensor","t":1733650000,"temp_gpu":54.2,"temp_room":22.1,"psu_amps":21.3,"light":0.3}
```

#### Status Response
```json
{"type":"status","uptime_s":3600,"relay1":"armed","relay2":"off","watchdog":"ok"}
```

#### Event
```json
{"type":"event","event":"estop_pressed","t":1733650000}
{"type":"event","event":"watchdog_timeout","t":1733650000}
{"type":"event","event":"temp_critical","sensor":"gpu","value":95.2}
```

#### Ack
```json
{"type":"ack","cmd":"arm","ok":true}
{"type":"ack","cmd":"kill","ok":true,"relay":1}
```

## Timing

- **Heartbeat interval**: 5 seconds (configurable)
- **Watchdog timeout**: 30 seconds (no heartbeat → action)
- **Sensor report interval**: 500ms (configurable)

## Error Handling

If Arduino receives malformed JSON:
```json
{"type":"error","msg":"parse_error"}
```

## Example Session

```
Ara → Arduino: {"cmd":"status"}
Arduino → Ara: {"type":"status","uptime_s":120,"relay1":"armed","watchdog":"ok"}

Ara → Arduino: {"cmd":"hb"}
Arduino → Ara: {"type":"ack","cmd":"hb","ok":true}

Arduino → Ara: {"type":"sensor","t":1733650100,"temp_gpu":65.3,"light":0.4}
Arduino → Ara: {"type":"sensor","t":1733650600,"temp_gpu":78.1,"light":0.4}
Arduino → Ara: {"type":"event","event":"temp_warning","sensor":"gpu","value":78.1}

Ara → Arduino: {"cmd":"kill","relay":1}
Arduino → Ara: {"type":"ack","cmd":"kill","ok":true,"relay":1}
Arduino → Ara: {"type":"event","event":"relay_off","relay":1}
```

## Safety Notes

1. **Watchdog is fail-safe**: If Ara dies, Arduino acts (power cycle or alert)
2. **E-STOP is physical**: Hardware button bypasses all software
3. **Protected relays**: Some relays can be marked "can't kill remotely"
4. **Logging**: Arduino logs all commands received (if space permits)
