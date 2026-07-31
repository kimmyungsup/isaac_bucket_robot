/*
  Arduino Nano continuous paint switch

  Wiring:
    - Switch: D2 <-> GND
    - LED: built-in LED (L)

  Serial protocol:
    - Sends the single ASCII byte 'A' repeatedly while the switch is held.
    - Sends nothing while the switch is released.
    - Baud rate: 9600
*/

const byte SWITCH_PIN = 2;
const byte LED_PIN = LED_BUILTIN;

const unsigned long DEBOUNCE_MS = 20;
const unsigned long SEND_INTERVAL_MS = 50;

bool rawPressed = false;
bool stablePressed = false;
unsigned long rawChangedAt = 0;
unsigned long lastSentAt = 0;

void setup() {
  pinMode(SWITCH_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  Serial.begin(9600);

  rawPressed = (digitalRead(SWITCH_PIN) == LOW);
  stablePressed = rawPressed;
  rawChangedAt = millis();

  if (stablePressed) {
    digitalWrite(LED_PIN, HIGH);
    Serial.write('A');
    lastSentAt = millis();
  }
}

void loop() {
  const unsigned long now = millis();
  const bool pressed = (digitalRead(SWITCH_PIN) == LOW);

  if (pressed != rawPressed) {
    rawPressed = pressed;
    rawChangedAt = now;
  }

  if (rawPressed != stablePressed &&
      now - rawChangedAt >= DEBOUNCE_MS) {
    stablePressed = rawPressed;
    digitalWrite(LED_PIN, stablePressed ? HIGH : LOW);

    // Send immediately after a confirmed press.
    if (stablePressed) {
      Serial.write('A');
      lastSentAt = now;
    }
  }

  // Keep sending the same one-byte signal for as long as the switch is held.
  if (stablePressed && now - lastSentAt >= SEND_INTERVAL_MS) {
    Serial.write('A');
    lastSentAt = now;
  }
}
