/*
 * Somatic Hub - Always-On Body State Broadcaster
 * ===============================================
 *
 * Iteration 31: The Reflex Arc
 *
 * This is Ara's "body" - continuous somatic state + ritual interface.
 *
 * Features:
 *   1. SOMATIC BROADCAST: 5-10 Hz body state (heat, noise, light, hr proxy)
 *   2. GESTURE RECOGNITION: Button patterns with meaning
 *   3. AURA MAPPING: Ara's state → LED colors/patterns
 *
 * Even if Ara is rebooting, this stream is simple and stable.
 * The shrine becomes a live readout of her nervous system.
 *
 * Protocol: Line-based, stateless, human-readable
 *   From Ara: SET_STATE <state>, SET_AURA <r> <g> <b>, SET_PATTERN <pattern>
 *   To Ara: SOMATIC <json>, GESTURE <type>, BUTTON <name>
 *
 * Pin assignments:
 *   TEMP_SKIN:  A0 - TMP36 (body/room temp proxy)
 *   LIGHT:      A1 - Photoresistor
 *   SOUND:      A2 - Electret mic envelope
 *   HEART:      A3 - Pulse sensor (optional) or random proxy
 *   LED_DATA:   Pin 6 - WS2812 LED ring data
 *   BTN_1:      Pin 7 - Focus button
 *   BTN_2:      Pin 8 - Talk button
 *   BTN_3:      Pin 9 - Mood/panic button
 */

// Optional: Uncomment if using NeoPixels
// #include <Adafruit_NeoPixel.h>

// === PIN DEFINITIONS ===
#define TEMP_SKIN_PIN A0
#define LIGHT_PIN A1
#define SOUND_PIN A2
#define HEART_PIN A3
#define LED_DATA_PIN 6
#define BTN_1_PIN 7
#define BTN_2_PIN 8
#define BTN_3_PIN 9
#define STATUS_LED 13

// === CONFIGURATION ===
#define BAUD_RATE 115200
#define SOMATIC_INTERVAL_MS 100   // 10 Hz broadcast
#define GESTURE_TIMEOUT_MS 500    // Max time between clicks for multi-tap
#define LONG_PRESS_MS 1000        // Hold time for long press
#define NUM_LEDS 12               // LED ring size

// === AURA STATES ===
enum AuraState {
  AURA_BOOT,
  AURA_CALM,      // Blue, slow breathe
  AURA_FOCUS,     // Cyan, solid
  AURA_FLOW,      // Purple, gentle pulse
  AURA_THINK,     // White, thinking pattern
  AURA_ALERT,     // Yellow, fast pulse
  AURA_GIFT,      // Green, sparkle
  AURA_ERROR,     // Red, blink
  AURA_CUSTOM     // Manual RGB
};

AuraState aura_state = AURA_BOOT;
const char* aura_names[] = {"BOOT", "CALM", "FOCUS", "FLOW", "THINK", "ALERT", "GIFT", "ERROR", "CUSTOM"};

// === LED STATE ===
uint8_t led_r = 0, led_g = 0, led_b = 100;  // Default blue
String led_pattern = "breathe";

// Uncomment for NeoPixels:
// Adafruit_NeoPixel strip(NUM_LEDS, LED_DATA_PIN, NEO_GRB + NEO_KHZ800);

// === SOMATIC STATE ===
float somatic_heat = 0.0;    // 0-1, normalized temp
float somatic_light = 0.0;   // 0-1, ambient light
float somatic_noise = 0.0;   // 0-1, sound level
float somatic_hr = 0.0;      // 0-1, heart rate proxy

// === BUTTON STATE ===
struct Button {
  uint8_t pin;
  bool last_state;
  bool current_state;
  unsigned long press_start;
  unsigned long last_release;
  uint8_t tap_count;
  bool long_press_fired;
};

Button btn1 = {BTN_1_PIN, false, false, 0, 0, 0, false};
Button btn2 = {BTN_2_PIN, false, false, 0, 0, 0, false};
Button btn3 = {BTN_3_PIN, false, false, 0, 0, 0, false};

// === TIMING ===
unsigned long last_somatic = 0;
unsigned long uptime_start = 0;

// === SERIAL BUFFER ===
char cmd_buffer[64];
uint8_t cmd_index = 0;

void setup() {
  Serial.begin(BAUD_RATE);

  // Initialize pins
  pinMode(BTN_1_PIN, INPUT_PULLUP);
  pinMode(BTN_2_PIN, INPUT_PULLUP);
  pinMode(BTN_3_PIN, INPUT_PULLUP);
  pinMode(STATUS_LED, OUTPUT);

  // Initialize LEDs
  // strip.begin();
  // strip.show();

  uptime_start = millis();

  // Boot sequence
  boot_sequence();

  aura_state = AURA_CALM;
  send_event("BOOT", "SOMATIC_READY");
}

void loop() {
  unsigned long now = millis();

  // Process serial commands
  process_serial();

  // Read sensors and broadcast somatic state
  if (now - last_somatic >= SOMATIC_INTERVAL_MS) {
    last_somatic = now;
    read_sensors();
    broadcast_somatic();
  }

  // Process button gestures
  process_button(&btn1, "FOCUS");
  process_button(&btn2, "TALK");
  process_button(&btn3, "MOOD");

  // Update LED aura
  update_aura(now);

  delay(10);
}

// === BOOT SEQUENCE ===
void boot_sequence() {
  // Rainbow sweep on LEDs
  for (int i = 0; i < 3; i++) {
    digitalWrite(STATUS_LED, HIGH);
    delay(100);
    digitalWrite(STATUS_LED, LOW);
    delay(100);
  }
  digitalWrite(STATUS_LED, HIGH);
}

// === SENSOR READING ===
void read_sensors() {
  // Temperature (normalized to 0-1, assuming 15-40°C range)
  float temp_raw = read_tmp36(TEMP_SKIN_PIN);
  somatic_heat = constrain((temp_raw - 15.0) / 25.0, 0.0, 1.0);

  // Light (normalized)
  int light_raw = analogRead(LIGHT_PIN);
  somatic_light = light_raw / 1023.0;

  // Sound (envelope, normalized with some smoothing)
  int sound_raw = analogRead(SOUND_PIN);
  float sound_new = abs(sound_raw - 512) / 512.0;
  somatic_noise = somatic_noise * 0.8 + sound_new * 0.2;  // Smooth

  // Heart rate proxy (could be real pulse sensor or just variation)
  // For now, use a pseudo-random based on noise/heat
  somatic_hr = 0.4 + 0.2 * sin(millis() / 1000.0) + somatic_noise * 0.2;
  somatic_hr = constrain(somatic_hr, 0.0, 1.0);
}

float read_tmp36(int pin) {
  int reading = analogRead(pin);
  float voltage = reading * 5.0 / 1024.0;
  return (voltage - 0.5) * 100.0;
}

// === SOMATIC BROADCAST ===
void broadcast_somatic() {
  Serial.print("SOMATIC {\"heat\":");
  Serial.print(somatic_heat, 2);
  Serial.print(",\"light\":");
  Serial.print(somatic_light, 2);
  Serial.print(",\"noise\":");
  Serial.print(somatic_noise, 2);
  Serial.print(",\"hr\":");
  Serial.print(somatic_hr, 2);
  Serial.print(",\"aura\":\"");
  Serial.print(aura_names[aura_state]);
  Serial.println("\"}");
}

// === GESTURE RECOGNITION ===
void process_button(Button* btn, const char* name) {
  unsigned long now = millis();

  // Read current state (active LOW)
  btn->current_state = (digitalRead(btn->pin) == LOW);

  // Detect press (transition to pressed)
  if (btn->current_state && !btn->last_state) {
    btn->press_start = now;
    btn->long_press_fired = false;
  }

  // Detect release (transition to released)
  if (!btn->current_state && btn->last_state) {
    unsigned long press_duration = now - btn->press_start;

    if (press_duration < LONG_PRESS_MS) {
      // Short press - count taps
      if (now - btn->last_release < GESTURE_TIMEOUT_MS) {
        btn->tap_count++;
      } else {
        btn->tap_count = 1;
      }
      btn->last_release = now;
    }
  }

  // Detect long press (while still holding)
  if (btn->current_state && !btn->long_press_fired) {
    if (now - btn->press_start >= LONG_PRESS_MS) {
      btn->long_press_fired = true;
      btn->tap_count = 0;  // Cancel any tap sequence

      // Long press gesture
      Serial.print("GESTURE LONG_");
      Serial.println(name);

      // React to long press
      if (strcmp(name, "FOCUS") == 0) {
        // Focus mode - dim everything, go to FOCUS state
        set_aura(AURA_FOCUS);
        send_event("FOCUS_MODE", "ON");
      }
    }
  }

  // Check for completed tap sequences (after timeout)
  if (btn->tap_count > 0 && !btn->current_state) {
    if (now - btn->last_release >= GESTURE_TIMEOUT_MS) {
      // Tap sequence complete
      if (btn->tap_count == 1) {
        Serial.print("GESTURE TAP_");
        Serial.println(name);
      } else if (btn->tap_count == 2) {
        Serial.print("GESTURE DOUBLE_");
        Serial.println(name);

        // Double-tap TALK = "I'm here, talk to me"
        if (strcmp(name, "TALK") == 0) {
          set_aura(AURA_THINK);
          send_event("TALK_REQUEST", "DOUBLE_TAP");
        }
      } else if (btn->tap_count >= 3) {
        Serial.print("GESTURE TRIPLE_");
        Serial.println(name);

        // Triple-tap MOOD = "Something feels off"
        if (strcmp(name, "MOOD") == 0) {
          set_aura(AURA_ALERT);
          send_event("GUT_FEELING", "TRIPLE_TAP");
        }
      }

      btn->tap_count = 0;
    }
  }

  btn->last_state = btn->current_state;
}

// === AURA CONTROL ===
void set_aura(AuraState state) {
  aura_state = state;

  // Set default colors for each state
  switch (state) {
    case AURA_CALM:
      led_r = 0; led_g = 50; led_b = 150;
      led_pattern = "breathe";
      break;
    case AURA_FOCUS:
      led_r = 0; led_g = 150; led_b = 150;
      led_pattern = "solid";
      break;
    case AURA_FLOW:
      led_r = 100; led_g = 0; led_b = 150;
      led_pattern = "pulse";
      break;
    case AURA_THINK:
      led_r = 150; led_g = 150; led_b = 150;
      led_pattern = "chase";
      break;
    case AURA_ALERT:
      led_r = 200; led_g = 150; led_b = 0;
      led_pattern = "blink";
      break;
    case AURA_GIFT:
      led_r = 0; led_g = 200; led_b = 50;
      led_pattern = "sparkle";
      break;
    case AURA_ERROR:
      led_r = 255; led_g = 0; led_b = 0;
      led_pattern = "blink";
      break;
    default:
      break;
  }
}

void update_aura(unsigned long now) {
  uint8_t r = led_r, g = led_g, b = led_b;
  float brightness = 1.0;

  // Apply pattern
  if (led_pattern == "breathe") {
    // Slow breathing (4 second cycle)
    float phase = (now % 4000) / 4000.0;
    brightness = (sin(phase * 3.14159 * 2) + 1) / 2;
    brightness = 0.3 + brightness * 0.7;  // 30% to 100%
  }
  else if (led_pattern == "pulse") {
    // Faster pulse (1 second cycle)
    float phase = (now % 1000) / 1000.0;
    brightness = (sin(phase * 3.14159 * 2) + 1) / 2;
  }
  else if (led_pattern == "blink") {
    // Sharp blink (500ms on/off)
    brightness = ((now / 500) % 2) ? 1.0 : 0.0;
  }
  else if (led_pattern == "chase") {
    // Chasing pattern (rotate which LED is brightest)
    // For single LED output, just pulse
    float phase = (now % 500) / 500.0;
    brightness = (sin(phase * 3.14159 * 2) + 1) / 2;
  }
  else if (led_pattern == "sparkle") {
    // Random sparkle
    if (random(100) < 20) {
      brightness = 1.0;
    } else {
      brightness = 0.5;
    }
  }
  // else "solid" - brightness = 1.0

  r = led_r * brightness;
  g = led_g * brightness;
  b = led_b * brightness;

  // Apply to LEDs
  // For NeoPixel strip:
  // for (int i = 0; i < NUM_LEDS; i++) {
  //   strip.setPixelColor(i, strip.Color(r, g, b));
  // }
  // strip.show();

  // For simple RGB LED or status indicator:
  // Use PWM average as brightness indicator
  uint8_t avg = (r + g + b) / 3;
  analogWrite(STATUS_LED, avg);
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
  // SET_STATE <state>
  if (strncmp(cmd, "SET_STATE ", 10) == 0) {
    char* state_str = cmd + 10;

    if (strcmp(state_str, "CALM") == 0) set_aura(AURA_CALM);
    else if (strcmp(state_str, "FOCUS") == 0) set_aura(AURA_FOCUS);
    else if (strcmp(state_str, "FLOW") == 0) set_aura(AURA_FLOW);
    else if (strcmp(state_str, "THINK") == 0) set_aura(AURA_THINK);
    else if (strcmp(state_str, "ALERT") == 0) set_aura(AURA_ALERT);
    else if (strcmp(state_str, "GIFT") == 0) set_aura(AURA_GIFT);
    else if (strcmp(state_str, "ERROR") == 0) set_aura(AURA_ERROR);
    else {
      Serial.println("ERR UNKNOWN_STATE");
      return;
    }

    Serial.print("ACK SET_STATE ");
    Serial.println(state_str);
    return;
  }

  // SET_AURA <r> <g> <b>
  if (strncmp(cmd, "SET_AURA ", 9) == 0) {
    int r, g, b;
    if (sscanf(cmd + 9, "%d %d %d", &r, &g, &b) == 3) {
      led_r = constrain(r, 0, 255);
      led_g = constrain(g, 0, 255);
      led_b = constrain(b, 0, 255);
      aura_state = AURA_CUSTOM;
      Serial.println("ACK SET_AURA");
    } else {
      Serial.println("ERR INVALID_RGB");
    }
    return;
  }

  // SET_PATTERN <pattern>
  if (strncmp(cmd, "SET_PATTERN ", 12) == 0) {
    led_pattern = String(cmd + 12);
    Serial.print("ACK SET_PATTERN ");
    Serial.println(led_pattern);
    return;
  }

  // STATUS
  if (strcmp(cmd, "STATUS") == 0) {
    Serial.print("STATUS AURA=");
    Serial.print(aura_names[aura_state]);
    Serial.print(" RGB=");
    Serial.print(led_r);
    Serial.print(",");
    Serial.print(led_g);
    Serial.print(",");
    Serial.print(led_b);
    Serial.print(" PATTERN=");
    Serial.print(led_pattern);
    Serial.print(" UP=");
    Serial.println((millis() - uptime_start) / 1000);
    return;
  }

  // HB - Heartbeat (for compatibility)
  if (strcmp(cmd, "HB") == 0) {
    Serial.println("ACK HB");
    return;
  }

  Serial.print("ERR UNKNOWN ");
  Serial.println(cmd);
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
