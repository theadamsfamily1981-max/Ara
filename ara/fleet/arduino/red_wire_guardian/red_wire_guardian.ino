/*
 * Red Wire Guardian - Out-of-Band Safety Controller
 * ==================================================
 *
 * Arduino-based safety system that:
 * - Controls power relays to critical systems
 * - Implements hardware watchdog (if Ara dies, take action)
 * - Provides E-STOP button input
 * - Reports status over serial
 *
 * Protocol: ara:// JSON over serial (115200 baud)
 *
 * Pin assignments (adjust for your setup):
 *   RELAY_1: Pin 2 - GPU rig power
 *   RELAY_2: Pin 3 - FPGA/SNN power
 *   RELAY_3: Pin 4 - Server power
 *   ESTOP:   Pin 5 - Emergency stop button (active LOW)
 *   STATUS_LED: Pin 13 - Built-in LED for status
 */

#include <ArduinoJson.h>

// === PIN DEFINITIONS ===
#define RELAY_1_PIN 2
#define RELAY_2_PIN 3
#define RELAY_3_PIN 4
#define ESTOP_PIN 5
#define STATUS_LED 13

// === CONFIGURATION ===
#define BAUD_RATE 115200
#define WATCHDOG_TIMEOUT_MS 30000  // 30 seconds without heartbeat
#define HEARTBEAT_LED_BLINK_MS 100
#define SENSOR_INTERVAL_MS 1000

// === STATE ===
bool relay_states[3] = {false, false, false};  // true = powered/armed
bool relay_armed[3] = {true, true, true};      // can be remotely controlled
bool relay_protected[3] = {false, false, true}; // relay 3 is protected (brainstem)

unsigned long last_heartbeat = 0;
unsigned long last_sensor_report = 0;
unsigned long uptime_start = 0;
bool watchdog_triggered = false;
bool estop_pressed = false;

// JSON buffer
StaticJsonDocument<256> doc;
char json_buffer[256];

void setup() {
  Serial.begin(BAUD_RATE);

  // Initialize pins
  pinMode(RELAY_1_PIN, OUTPUT);
  pinMode(RELAY_2_PIN, OUTPUT);
  pinMode(RELAY_3_PIN, OUTPUT);
  pinMode(ESTOP_PIN, INPUT_PULLUP);
  pinMode(STATUS_LED, OUTPUT);

  // Start with relays off (safe default)
  digitalWrite(RELAY_1_PIN, LOW);
  digitalWrite(RELAY_2_PIN, LOW);
  digitalWrite(RELAY_3_PIN, LOW);

  uptime_start = millis();
  last_heartbeat = millis();

  // Announce ourselves
  send_event("boot", "guardian_ready");
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.length() > 0) {
      process_command(line);
    }
  }

  // Check E-STOP button
  check_estop();

  // Check watchdog
  check_watchdog();

  // Blink status LED
  update_status_led();

  // Small delay
  delay(10);
}

void process_command(String& line) {
  DeserializationError error = deserializeJson(doc, line);

  if (error) {
    send_error("parse_error");
    return;
  }

  const char* cmd = doc["cmd"];
  if (!cmd) {
    send_error("no_cmd");
    return;
  }

  if (strcmp(cmd, "hb") == 0) {
    // Heartbeat - reset watchdog timer
    last_heartbeat = millis();
    watchdog_triggered = false;
    send_ack("hb", true);

  } else if (strcmp(cmd, "arm") == 0) {
    int relay = doc["relay"] | 0;
    if (relay >= 1 && relay <= 3) {
      relay_armed[relay-1] = true;
      set_relay(relay, true);
      send_ack("arm", true);
    } else {
      send_ack("arm", false);
    }

  } else if (strcmp(cmd, "disarm") == 0) {
    int relay = doc["relay"] | 0;
    if (relay >= 1 && relay <= 3) {
      relay_armed[relay-1] = false;
      send_ack("disarm", true);
    } else {
      send_ack("disarm", false);
    }

  } else if (strcmp(cmd, "kill") == 0) {
    int relay = doc["relay"] | 0;
    if (relay >= 1 && relay <= 3) {
      if (relay_protected[relay-1]) {
        send_ack("kill", false);  // Protected relay
        send_event("protected_refused", String(relay).c_str());
      } else {
        set_relay(relay, false);
        relay_armed[relay-1] = false;  // Stay off until rearmed
        send_ack("kill", true);
      }
    } else {
      send_ack("kill", false);
    }

  } else if (strcmp(cmd, "cycle") == 0) {
    int relay = doc["relay"] | 0;
    int off_ms = doc["off_ms"] | 5000;
    if (relay >= 1 && relay <= 3 && !relay_protected[relay-1]) {
      power_cycle(relay, off_ms);
      send_ack("cycle", true);
    } else {
      send_ack("cycle", false);
    }

  } else if (strcmp(cmd, "status") == 0) {
    send_status();

  } else {
    send_error("unknown_cmd");
  }
}

void set_relay(int relay, bool on) {
  int pin = RELAY_1_PIN + relay - 1;
  digitalWrite(pin, on ? HIGH : LOW);
  relay_states[relay-1] = on;

  // Send event
  if (on) {
    send_event("relay_on", String(relay).c_str());
  } else {
    send_event("relay_off", String(relay).c_str());
  }
}

void power_cycle(int relay, int off_ms) {
  set_relay(relay, false);
  delay(off_ms);
  set_relay(relay, true);
}

void check_estop() {
  bool pressed = (digitalRead(ESTOP_PIN) == LOW);

  if (pressed && !estop_pressed) {
    // E-STOP just pressed
    estop_pressed = true;
    send_event("estop_pressed", "");

    // Kill all non-protected relays
    for (int i = 0; i < 3; i++) {
      if (!relay_protected[i]) {
        set_relay(i+1, false);
        relay_armed[i] = false;
      }
    }
  } else if (!pressed && estop_pressed) {
    // E-STOP released
    estop_pressed = false;
    send_event("estop_released", "");
  }
}

void check_watchdog() {
  unsigned long now = millis();

  if (now - last_heartbeat > WATCHDOG_TIMEOUT_MS) {
    if (!watchdog_triggered) {
      watchdog_triggered = true;
      send_event("watchdog_timeout", "");

      // Watchdog action: power cycle relay 1 (main rig)
      // Modify this based on your policy
      if (relay_armed[0] && !relay_protected[0]) {
        power_cycle(1, 5000);
        send_event("watchdog_cycle", "relay_1");
      }
    }
  }
}

void update_status_led() {
  // Blink pattern indicates state
  unsigned long now = millis();

  if (estop_pressed) {
    // Fast blink when E-STOP pressed
    digitalWrite(STATUS_LED, (now / 100) % 2);
  } else if (watchdog_triggered) {
    // Slow blink when watchdog triggered
    digitalWrite(STATUS_LED, (now / 500) % 2);
  } else {
    // Solid when healthy
    digitalWrite(STATUS_LED, HIGH);
  }
}

void send_status() {
  doc.clear();
  doc["type"] = "status";
  doc["uptime_s"] = (millis() - uptime_start) / 1000;
  doc["relay1"] = relay_states[0] ? "on" : "off";
  doc["relay2"] = relay_states[1] ? "on" : "off";
  doc["relay3"] = relay_states[2] ? "on" : "off";
  doc["armed1"] = relay_armed[0];
  doc["armed2"] = relay_armed[1];
  doc["armed3"] = relay_armed[2];
  doc["watchdog"] = watchdog_triggered ? "triggered" : "ok";
  doc["estop"] = estop_pressed ? "pressed" : "released";

  serializeJson(doc, json_buffer);
  Serial.println(json_buffer);
}

void send_ack(const char* cmd, bool ok) {
  doc.clear();
  doc["type"] = "ack";
  doc["cmd"] = cmd;
  doc["ok"] = ok;

  serializeJson(doc, json_buffer);
  Serial.println(json_buffer);
}

void send_event(const char* event, const char* detail) {
  doc.clear();
  doc["type"] = "event";
  doc["event"] = event;
  doc["t"] = millis() / 1000;
  if (strlen(detail) > 0) {
    doc["detail"] = detail;
  }

  serializeJson(doc, json_buffer);
  Serial.println(json_buffer);
}

void send_error(const char* msg) {
  doc.clear();
  doc["type"] = "error";
  doc["msg"] = msg;

  serializeJson(doc, json_buffer);
  Serial.println(json_buffer);
}
