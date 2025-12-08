/*
 * Somatic Node - Ara's Body Sensors + Mood Ring
 * ==============================================
 *
 * Arduino-based sensor hub that:
 * - Reads temperature sensors (DS18B20 or TMP36)
 * - Reads ambient light level
 * - Reads current sensors on PSU rails
 * - Controls LED ring/strip for status display
 * - Provides button inputs for user interaction
 *
 * Protocol: ara:// JSON over serial (115200 baud)
 *
 * Pin assignments (adjust for your setup):
 *   TEMP_GPU:    A0 - TMP36 near GPU
 *   TEMP_ROOM:   A1 - TMP36 for room temp
 *   LIGHT:       A2 - Photoresistor
 *   CURRENT:     A3 - ACS712 current sensor
 *   LED_DATA:    Pin 6 - WS2812 LED strip data
 *   BTN_FOCUS:   Pin 7 - Focus mode button
 *   BTN_TALK:    Pin 8 - "Talk to me" button
 *   BTN_MOOD:    Pin 9 - "Something's wrong" button
 */

#include <ArduinoJson.h>
// Uncomment if using NeoPixels:
// #include <Adafruit_NeoPixel.h>

// === PIN DEFINITIONS ===
#define TEMP_GPU_PIN A0
#define TEMP_ROOM_PIN A1
#define LIGHT_PIN A2
#define CURRENT_PIN A3
#define LED_DATA_PIN 6
#define BTN_FOCUS_PIN 7
#define BTN_TALK_PIN 8
#define BTN_MOOD_PIN 9
#define STATUS_LED 13

// === CONFIGURATION ===
#define BAUD_RATE 115200
#define SENSOR_INTERVAL_MS 500
#define DEBOUNCE_MS 50
#define NUM_LEDS 12  // Adjust for your LED ring

// === STATE ===
unsigned long last_sensor_report = 0;
unsigned long uptime_start = 0;

// Button states
bool btn_focus_last = false;
bool btn_talk_last = false;
bool btn_mood_last = false;
unsigned long btn_focus_time = 0;
unsigned long btn_talk_time = 0;
unsigned long btn_mood_time = 0;

// LED state
uint8_t led_r = 0, led_g = 50, led_b = 100;  // Default: calm blue
String led_pattern = "solid";

// JSON buffer
StaticJsonDocument<256> doc;
char json_buffer[256];

// Uncomment if using NeoPixels:
// Adafruit_NeoPixel strip(NUM_LEDS, LED_DATA_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  Serial.begin(BAUD_RATE);

  // Initialize pins
  pinMode(TEMP_GPU_PIN, INPUT);
  pinMode(TEMP_ROOM_PIN, INPUT);
  pinMode(LIGHT_PIN, INPUT);
  pinMode(CURRENT_PIN, INPUT);

  pinMode(BTN_FOCUS_PIN, INPUT_PULLUP);
  pinMode(BTN_TALK_PIN, INPUT_PULLUP);
  pinMode(BTN_MOOD_PIN, INPUT_PULLUP);

  pinMode(STATUS_LED, OUTPUT);

  // Initialize LEDs
  // strip.begin();
  // strip.show();

  uptime_start = millis();

  // Announce ourselves
  send_event("boot", "somatic_ready");
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

  // Check buttons
  check_buttons();

  // Report sensors periodically
  unsigned long now = millis();
  if (now - last_sensor_report >= SENSOR_INTERVAL_MS) {
    last_sensor_report = now;
    send_sensor_reading();
  }

  // Update LED pattern
  update_leds();

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

  if (strcmp(cmd, "led") == 0) {
    // Set LED color/pattern
    if (doc.containsKey("r")) {
      led_r = doc["r"] | 0;
      led_g = doc["g"] | 0;
      led_b = doc["b"] | 0;
    }
    if (doc.containsKey("pattern")) {
      led_pattern = doc["pattern"].as<String>();
    }
    if (doc.containsKey("color")) {
      String color = doc["color"].as<String>();
      if (color == "red") { led_r = 255; led_g = 0; led_b = 0; }
      else if (color == "green") { led_r = 0; led_g = 255; led_b = 0; }
      else if (color == "blue") { led_r = 0; led_g = 0; led_b = 255; }
      else if (color == "yellow") { led_r = 255; led_g = 255; led_b = 0; }
      else if (color == "purple") { led_r = 128; led_g = 0; led_b = 255; }
      else if (color == "cyan") { led_r = 0; led_g = 255; led_b = 255; }
      else if (color == "white") { led_r = 255; led_g = 255; led_b = 255; }
      else if (color == "off") { led_r = 0; led_g = 0; led_b = 0; }
    }
    send_ack("led", true);

  } else if (strcmp(cmd, "status") == 0) {
    send_status();

  } else if (strcmp(cmd, "hb") == 0) {
    // Heartbeat (for compatibility with guardian protocol)
    send_ack("hb", true);

  } else {
    send_error("unknown_cmd");
  }
}

float read_temp_c(int pin) {
  // TMP36 conversion: voltage = reading * 5.0 / 1024
  // temp_c = (voltage - 0.5) * 100
  int reading = analogRead(pin);
  float voltage = reading * 5.0 / 1024.0;
  float temp_c = (voltage - 0.5) * 100.0;
  return temp_c;
}

float read_light() {
  // Photoresistor: 0 = dark, 1 = bright
  int reading = analogRead(LIGHT_PIN);
  return reading / 1023.0;
}

float read_current_amps() {
  // ACS712 5A module: 2.5V = 0A, 185mV/A
  // Adjust for your specific sensor
  int reading = analogRead(CURRENT_PIN);
  float voltage = reading * 5.0 / 1024.0;
  float amps = (voltage - 2.5) / 0.185;
  return abs(amps);  // Return absolute value
}

void send_sensor_reading() {
  float temp_gpu = read_temp_c(TEMP_GPU_PIN);
  float temp_room = read_temp_c(TEMP_ROOM_PIN);
  float light = read_light();
  float amps = read_current_amps();

  doc.clear();
  doc["type"] = "sensor";
  doc["t"] = millis() / 1000;
  doc["temp_gpu"] = round(temp_gpu * 10) / 10.0;
  doc["temp_room"] = round(temp_room * 10) / 10.0;
  doc["light"] = round(light * 100) / 100.0;
  doc["psu_amps"] = round(amps * 10) / 10.0;

  serializeJson(doc, json_buffer);
  Serial.println(json_buffer);

  // Check for warnings
  if (temp_gpu > 80.0) {
    send_event("temp_warning", "gpu");
  }
  if (temp_gpu > 95.0) {
    send_event("temp_critical", "gpu");
  }
}

void check_buttons() {
  unsigned long now = millis();

  // Focus button
  bool focus_pressed = (digitalRead(BTN_FOCUS_PIN) == LOW);
  if (focus_pressed && !btn_focus_last && (now - btn_focus_time > DEBOUNCE_MS)) {
    btn_focus_time = now;
    send_event("button", "focus");
  }
  btn_focus_last = focus_pressed;

  // Talk button
  bool talk_pressed = (digitalRead(BTN_TALK_PIN) == LOW);
  if (talk_pressed && !btn_talk_last && (now - btn_talk_time > DEBOUNCE_MS)) {
    btn_talk_time = now;
    send_event("button", "talk");
  }
  btn_talk_last = talk_pressed;

  // Mood button
  bool mood_pressed = (digitalRead(BTN_MOOD_PIN) == LOW);
  if (mood_pressed && !btn_mood_last && (now - btn_mood_time > DEBOUNCE_MS)) {
    btn_mood_time = now;
    send_event("button", "mood");
  }
  btn_mood_last = mood_pressed;
}

void update_leds() {
  // Update LED strip based on pattern and color
  // Uncomment and adjust for your LED setup

  unsigned long now = millis();
  uint8_t r = led_r, g = led_g, b = led_b;

  if (led_pattern == "pulse") {
    // Pulsing brightness
    float phase = (now % 2000) / 2000.0;
    float brightness = (sin(phase * 3.14159 * 2) + 1) / 2;
    r = led_r * brightness;
    g = led_g * brightness;
    b = led_b * brightness;
  } else if (led_pattern == "blink") {
    // Blinking
    if ((now / 500) % 2 == 0) {
      r = g = b = 0;
    }
  } else if (led_pattern == "rainbow") {
    // Rainbow cycle (simplified)
    int hue = (now / 10) % 256;
    // HSV to RGB conversion would go here
  }

  // Apply to LED strip
  // for (int i = 0; i < NUM_LEDS; i++) {
  //   strip.setPixelColor(i, strip.Color(r, g, b));
  // }
  // strip.show();

  // For now, just use status LED
  analogWrite(STATUS_LED, (r + g + b) / 3);
}

void send_status() {
  doc.clear();
  doc["type"] = "status";
  doc["uptime_s"] = (millis() - uptime_start) / 1000;
  doc["led_r"] = led_r;
  doc["led_g"] = led_g;
  doc["led_b"] = led_b;
  doc["led_pattern"] = led_pattern;

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
