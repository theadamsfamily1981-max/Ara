/*
 * Spinal Cord - Autonomic Safety Controller
 * ==========================================
 *
 * Iteration 31: The Reflex Arc
 *
 * This is NOT a dumb relay board. It's a spinal cord with reflexes.
 *
 * Reflexes that fire WITHOUT the brain (PC/GPU):
 *   1. THERMAL REFLEX: Temp sensors → threshold → fan/relay action
 *   2. WATCHDOG REFLEX: No heartbeat + rising temp → graceful shutdown
 *   3. FAULT LATCH: Critical event → stay in FAULT until physical reset
 *
 * Even if Linux is wedged, the LLM is hallucinating, or USB is dead,
 * these reflexes keep running in microcontroller time (milliseconds).
 *
 * Protocol: Line-based, stateless, human-readable
 *   From Ara: HB, ARM, DISARM, SET_STATE <state>, SET_FAN <0-255>
 *   To Ara: FAULT <reason>, HB_LOST <seconds>, TEMP <zone> <value>, STATUS ...
 *
 * Pin assignments (adjust for your hardware):
 *   TEMP_GPU:     A0 - TMP36 near GPU
 *   TEMP_PSU:     A1 - TMP36 near PSU
 *   TEMP_HOTSPOT: A2 - TMP36 at hottest point
 *   FAN_PWM:      Pin 3 - PWM output to fan controller
 *   RELAY_GPU:    Pin 4 - GPU power rail
 *   RELAY_MAIN:   Pin 5 - Main system power
 *   RELAY_AUX:    Pin 6 - Auxiliary (miners, etc.)
 *   ESTOP_BTN:    Pin 7 - Physical E-STOP button
 *   RESET_BTN:    Pin 8 - Physical fault reset button
 *   LED_GREEN:    Pin 9 - Status OK
 *   LED_AMBER:    Pin 10 - Warning
 *   LED_RED:      Pin 11 - Fault
 *   BUZZER:       Pin 12 - Audible alarm
 */

// === PIN DEFINITIONS ===
#define TEMP_GPU_PIN A0
#define TEMP_PSU_PIN A1
#define TEMP_HOTSPOT_PIN A2
#define FAN_PWM_PIN 3
#define RELAY_GPU_PIN 4
#define RELAY_MAIN_PIN 5
#define RELAY_AUX_PIN 6
#define ESTOP_BTN_PIN 7
#define RESET_BTN_PIN 8
#define LED_GREEN_PIN 9
#define LED_AMBER_PIN 10
#define LED_RED_PIN 11
#define BUZZER_PIN 12

// === THERMAL THRESHOLDS (Celsius) ===
// These are HARD-CODED. No serial command can override them.
#define TEMP_WARN_GPU 75.0
#define TEMP_CRIT_GPU 90.0
#define TEMP_WARN_PSU 55.0
#define TEMP_CRIT_PSU 70.0
#define TEMP_WARN_HOTSPOT 80.0
#define TEMP_CRIT_HOTSPOT 95.0

// === TIMING ===
#define BAUD_RATE 115200
#define HEARTBEAT_TIMEOUT_MS 5000     // 5 seconds without HB = concern
#define HEARTBEAT_CRITICAL_MS 15000   // 15 seconds = take action
#define TEMP_READ_INTERVAL_MS 250     // Read temps 4x/second
#define STATUS_REPORT_INTERVAL_MS 1000 // Report to Ara 1x/second
#define GRACEFUL_SHUTDOWN_DELAY_MS 3000 // Wait between relay drops

// === STATE MACHINE ===
enum SpinalState {
  STATE_BOOT,      // Just powered on
  STATE_DISARMED,  // Not monitoring (safe for maintenance)
  STATE_ARMED,     // Normal operation, reflexes active
  STATE_WARNING,   // Threshold exceeded, fans maxed
  STATE_SHUTDOWN,  // Graceful shutdown in progress
  STATE_FAULT      // Latched fault, requires physical reset
};

SpinalState current_state = STATE_BOOT;
const char* state_names[] = {"BOOT", "DISARMED", "ARMED", "WARNING", "SHUTDOWN", "FAULT"};

// === FAULT CODES ===
enum FaultCode {
  FAULT_NONE = 0,
  FAULT_TEMP_GPU,
  FAULT_TEMP_PSU,
  FAULT_TEMP_HOTSPOT,
  FAULT_HEARTBEAT,
  FAULT_ESTOP
};

FaultCode fault_code = FAULT_NONE;
const char* fault_names[] = {"NONE", "TEMP_GPU_OVER", "TEMP_PSU_OVER", "TEMP_HOTSPOT_OVER", "HB_LOST", "ESTOP"};

// === RUNTIME STATE ===
unsigned long last_heartbeat = 0;
unsigned long last_temp_read = 0;
unsigned long last_status_report = 0;
unsigned long shutdown_started = 0;
unsigned long uptime_start = 0;

float temp_gpu = 0;
float temp_psu = 0;
float temp_hotspot = 0;

bool relay_gpu_on = true;
bool relay_main_on = true;
bool relay_aux_on = true;
uint8_t fan_speed = 128;  // 50% default

// Serial buffer
char cmd_buffer[64];
uint8_t cmd_index = 0;

void setup() {
  Serial.begin(BAUD_RATE);

  // Initialize outputs
  pinMode(FAN_PWM_PIN, OUTPUT);
  pinMode(RELAY_GPU_PIN, OUTPUT);
  pinMode(RELAY_MAIN_PIN, OUTPUT);
  pinMode(RELAY_AUX_PIN, OUTPUT);
  pinMode(LED_GREEN_PIN, OUTPUT);
  pinMode(LED_AMBER_PIN, OUTPUT);
  pinMode(LED_RED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  // Initialize inputs
  pinMode(ESTOP_BTN_PIN, INPUT_PULLUP);
  pinMode(RESET_BTN_PIN, INPUT_PULLUP);

  // Start with relays ON (power through)
  digitalWrite(RELAY_GPU_PIN, HIGH);
  digitalWrite(RELAY_MAIN_PIN, HIGH);
  digitalWrite(RELAY_AUX_PIN, HIGH);

  // Initial fan speed
  analogWrite(FAN_PWM_PIN, fan_speed);

  uptime_start = millis();
  last_heartbeat = millis();

  // Boot sequence
  boot_sequence();

  current_state = STATE_DISARMED;
  send_event("BOOT", "SPINAL_READY");
}

void loop() {
  unsigned long now = millis();

  // Always check physical buttons (even in FAULT)
  check_physical_buttons();

  // Read temps at interval
  if (now - last_temp_read >= TEMP_READ_INTERVAL_MS) {
    last_temp_read = now;
    read_temperatures();

    // Thermal reflex (only when armed)
    if (current_state == STATE_ARMED || current_state == STATE_WARNING) {
      thermal_reflex();
    }
  }

  // Heartbeat watchdog (only when armed)
  if (current_state == STATE_ARMED || current_state == STATE_WARNING) {
    heartbeat_watchdog(now);
  }

  // Process serial commands
  process_serial();

  // Status report at interval
  if (now - last_status_report >= STATUS_REPORT_INTERVAL_MS) {
    last_status_report = now;
    send_status();
  }

  // Update LEDs
  update_leds();

  // Handle shutdown sequence
  if (current_state == STATE_SHUTDOWN) {
    graceful_shutdown_tick(now);
  }

  delay(10);
}

// === BOOT SEQUENCE ===
void boot_sequence() {
  // LED test
  digitalWrite(LED_RED_PIN, HIGH);
  delay(200);
  digitalWrite(LED_RED_PIN, LOW);
  digitalWrite(LED_AMBER_PIN, HIGH);
  delay(200);
  digitalWrite(LED_AMBER_PIN, LOW);
  digitalWrite(LED_GREEN_PIN, HIGH);
  delay(200);

  // Short beep
  tone(BUZZER_PIN, 2000, 100);
}

// === TEMPERATURE READING ===
float read_tmp36(int pin) {
  int reading = analogRead(pin);
  float voltage = reading * 5.0 / 1024.0;
  return (voltage - 0.5) * 100.0;
}

void read_temperatures() {
  temp_gpu = read_tmp36(TEMP_GPU_PIN);
  temp_psu = read_tmp36(TEMP_PSU_PIN);
  temp_hotspot = read_tmp36(TEMP_HOTSPOT_PIN);
}

// === THERMAL REFLEX ===
// This is the core autonomic response. No PC needed.
void thermal_reflex() {
  bool warning = false;
  bool critical = false;
  FaultCode crit_fault = FAULT_NONE;

  // Check GPU temp
  if (temp_gpu >= TEMP_CRIT_GPU) {
    critical = true;
    crit_fault = FAULT_TEMP_GPU;
  } else if (temp_gpu >= TEMP_WARN_GPU) {
    warning = true;
  }

  // Check PSU temp
  if (temp_psu >= TEMP_CRIT_PSU) {
    critical = true;
    crit_fault = FAULT_TEMP_PSU;
  } else if (temp_psu >= TEMP_WARN_PSU) {
    warning = true;
  }

  // Check hotspot
  if (temp_hotspot >= TEMP_CRIT_HOTSPOT) {
    critical = true;
    crit_fault = FAULT_TEMP_HOTSPOT;
  } else if (temp_hotspot >= TEMP_WARN_HOTSPOT) {
    warning = true;
  }

  // Act on thresholds
  if (critical) {
    // CRITICAL: Immediate action
    enter_fault(crit_fault);
  } else if (warning && current_state != STATE_WARNING) {
    // WARNING: Max fans, alert
    enter_warning();
  } else if (!warning && current_state == STATE_WARNING) {
    // Recovered from warning
    current_state = STATE_ARMED;
    fan_speed = 128;  // Back to normal
    analogWrite(FAN_PWM_PIN, fan_speed);
    send_event("RECOVERED", "TEMP_NORMAL");
  }
}

// === HEARTBEAT WATCHDOG ===
void heartbeat_watchdog(unsigned long now) {
  unsigned long since_hb = now - last_heartbeat;

  if (since_hb >= HEARTBEAT_CRITICAL_MS) {
    // Critical: No heartbeat AND we're armed
    // Check if temps are also rising (lockup indicator)
    if (temp_gpu > TEMP_WARN_GPU || temp_hotspot > TEMP_WARN_HOTSPOT) {
      // System is likely locked up and overheating
      send_event("HB_LOST", String(since_hb / 1000).c_str());
      enter_fault(FAULT_HEARTBEAT);
    }
  } else if (since_hb >= HEARTBEAT_TIMEOUT_MS) {
    // Warning: Heartbeat late
    if (current_state == STATE_ARMED) {
      send_event("HB_LATE", String(since_hb / 1000).c_str());
    }
  }
}

// === STATE TRANSITIONS ===
void enter_warning() {
  current_state = STATE_WARNING;

  // Max fans
  fan_speed = 255;
  analogWrite(FAN_PWM_PIN, fan_speed);

  // Amber LED
  digitalWrite(LED_GREEN_PIN, LOW);
  digitalWrite(LED_AMBER_PIN, HIGH);

  // Warning beep
  tone(BUZZER_PIN, 1500, 500);

  send_event("WARNING", "THERMAL");
}

void enter_fault(FaultCode code) {
  fault_code = code;

  // Start graceful shutdown
  current_state = STATE_SHUTDOWN;
  shutdown_started = millis();

  // Max fans, red LED, alarm
  fan_speed = 255;
  analogWrite(FAN_PWM_PIN, fan_speed);
  digitalWrite(LED_GREEN_PIN, LOW);
  digitalWrite(LED_AMBER_PIN, LOW);
  digitalWrite(LED_RED_PIN, HIGH);

  // Alarm pattern
  tone(BUZZER_PIN, 3000, 1000);

  send_event("FAULT", fault_names[code]);
}

void graceful_shutdown_tick(unsigned long now) {
  unsigned long elapsed = now - shutdown_started;

  // Sequence: GPU first, wait, then main
  if (elapsed < GRACEFUL_SHUTDOWN_DELAY_MS) {
    // Phase 1: Kill GPU
    if (relay_gpu_on) {
      digitalWrite(RELAY_GPU_PIN, LOW);
      relay_gpu_on = false;
      send_event("RELAY_OFF", "GPU");
    }
  } else if (elapsed < GRACEFUL_SHUTDOWN_DELAY_MS * 2) {
    // Phase 2: Kill aux
    if (relay_aux_on) {
      digitalWrite(RELAY_AUX_PIN, LOW);
      relay_aux_on = false;
      send_event("RELAY_OFF", "AUX");
    }
  } else {
    // Phase 3: Kill main (optional - may want to keep for serial comms)
    // For now, enter latched FAULT state
    current_state = STATE_FAULT;
    noTone(BUZZER_PIN);

    // Periodic reminder beep
    send_event("SHUTDOWN_COMPLETE", fault_names[fault_code]);
  }
}

// === PHYSICAL BUTTONS ===
void check_physical_buttons() {
  // E-STOP: Always active
  if (digitalRead(ESTOP_BTN_PIN) == LOW) {
    if (current_state != STATE_FAULT) {
      send_event("ESTOP", "PRESSED");
      enter_fault(FAULT_ESTOP);
    }
  }

  // RESET: Only works in FAULT state
  if (digitalRead(RESET_BTN_PIN) == LOW) {
    if (current_state == STATE_FAULT) {
      delay(50);  // Debounce
      if (digitalRead(RESET_BTN_PIN) == LOW) {
        reset_from_fault();
      }
    }
  }
}

void reset_from_fault() {
  // Re-enable relays
  digitalWrite(RELAY_GPU_PIN, HIGH);
  digitalWrite(RELAY_MAIN_PIN, HIGH);
  digitalWrite(RELAY_AUX_PIN, HIGH);
  relay_gpu_on = true;
  relay_main_on = true;
  relay_aux_on = true;

  // Reset state
  fault_code = FAULT_NONE;
  current_state = STATE_DISARMED;  // Require explicit ARM
  fan_speed = 128;
  analogWrite(FAN_PWM_PIN, fan_speed);

  // Clear alarms
  noTone(BUZZER_PIN);
  digitalWrite(LED_RED_PIN, LOW);

  last_heartbeat = millis();

  send_event("RESET", "MANUAL");
}

// === SERIAL COMMAND PROCESSING ===
void process_serial() {
  while (Serial.available()) {
    char c = Serial.read();

    if (c == '\n' || c == '\r') {
      if (cmd_index > 0) {
        cmd_buffer[cmd_index] = '\0';
        execute_command(cmd_buffer);
        cmd_index = 0;
      }
    } else if (cmd_index < sizeof(cmd_buffer) - 1) {
      cmd_buffer[cmd_index++] = c;
    }
  }
}

void execute_command(char* cmd) {
  // HB - Heartbeat
  if (strcmp(cmd, "HB") == 0) {
    last_heartbeat = millis();
    Serial.println("ACK HB");
    return;
  }

  // ARM - Enable reflexes
  if (strcmp(cmd, "ARM") == 0) {
    if (current_state == STATE_DISARMED) {
      current_state = STATE_ARMED;
      last_heartbeat = millis();
      Serial.println("ACK ARM");
      send_event("ARMED", "OK");
    } else {
      Serial.println("ERR ALREADY_ARMED_OR_FAULT");
    }
    return;
  }

  // DISARM - Disable reflexes (for maintenance)
  if (strcmp(cmd, "DISARM") == 0) {
    if (current_state == STATE_ARMED || current_state == STATE_WARNING) {
      current_state = STATE_DISARMED;
      Serial.println("ACK DISARM");
      send_event("DISARMED", "OK");
    } else {
      Serial.println("ERR NOT_ARMED");
    }
    return;
  }

  // SET_FAN <0-255>
  if (strncmp(cmd, "SET_FAN ", 8) == 0) {
    int val = atoi(cmd + 8);
    if (val >= 0 && val <= 255 && current_state != STATE_FAULT) {
      fan_speed = val;
      analogWrite(FAN_PWM_PIN, fan_speed);
      Serial.print("ACK SET_FAN ");
      Serial.println(fan_speed);
    } else {
      Serial.println("ERR INVALID_FAN");
    }
    return;
  }

  // STATUS - Request full status
  if (strcmp(cmd, "STATUS") == 0) {
    send_status();
    return;
  }

  // Unknown command
  Serial.print("ERR UNKNOWN ");
  Serial.println(cmd);
}

// === STATUS REPORTING ===
void send_status() {
  Serial.print("STATUS ");
  Serial.print(state_names[current_state]);
  Serial.print(" FAULT=");
  Serial.print(fault_names[fault_code]);
  Serial.print(" GPU=");
  Serial.print(temp_gpu, 1);
  Serial.print(" PSU=");
  Serial.print(temp_psu, 1);
  Serial.print(" HOT=");
  Serial.print(temp_hotspot, 1);
  Serial.print(" FAN=");
  Serial.print(fan_speed);
  Serial.print(" RELAYS=");
  Serial.print(relay_gpu_on ? "G" : "g");
  Serial.print(relay_main_on ? "M" : "m");
  Serial.print(relay_aux_on ? "A" : "a");
  Serial.print(" UP=");
  Serial.println((millis() - uptime_start) / 1000);
}

void send_event(const char* event, const char* detail) {
  Serial.print("EVENT ");
  Serial.print(event);
  if (strlen(detail) > 0) {
    Serial.print(" ");
    Serial.print(detail);
  }
  Serial.println();
}

// === LED UPDATE ===
void update_leds() {
  unsigned long now = millis();

  switch (current_state) {
    case STATE_BOOT:
    case STATE_DISARMED:
      // Slow blink green
      digitalWrite(LED_GREEN_PIN, (now / 1000) % 2);
      digitalWrite(LED_AMBER_PIN, LOW);
      digitalWrite(LED_RED_PIN, LOW);
      break;

    case STATE_ARMED:
      // Solid green
      digitalWrite(LED_GREEN_PIN, HIGH);
      digitalWrite(LED_AMBER_PIN, LOW);
      digitalWrite(LED_RED_PIN, LOW);
      break;

    case STATE_WARNING:
      // Solid amber
      digitalWrite(LED_GREEN_PIN, LOW);
      digitalWrite(LED_AMBER_PIN, HIGH);
      digitalWrite(LED_RED_PIN, LOW);
      break;

    case STATE_SHUTDOWN:
      // Fast blink red
      digitalWrite(LED_GREEN_PIN, LOW);
      digitalWrite(LED_AMBER_PIN, LOW);
      digitalWrite(LED_RED_PIN, (now / 200) % 2);
      break;

    case STATE_FAULT:
      // Solid red + periodic beep
      digitalWrite(LED_GREEN_PIN, LOW);
      digitalWrite(LED_AMBER_PIN, LOW);
      digitalWrite(LED_RED_PIN, HIGH);

      // Reminder beep every 5 seconds
      if ((now / 5000) % 2 == 0 && (now % 5000) < 100) {
        tone(BUZZER_PIN, 2000, 100);
      }
      break;
  }
}
