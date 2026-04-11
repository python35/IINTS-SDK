/*
  IINTS UNO Q Supervisor Bridge

  Serial protocol:
  - OK
  - OVERRIDE
  - CRITICAL

  Use this sketch on the STM32 side of an Arduino UNO Q while the Linux side
  runs the IINTS digital patient runtime.
*/

// On some UNO Q boards the built-in status LED appears blue instead of green.
// The SDK docs refer to this as the "status LED" rather than assuming a color.
const int GREEN_LED_PIN = LED_BUILTIN;
const int RED_LED_PIN = 6;
const int BUZZER_PIN = 9;

String incomingLine = "";

void resetOutputs() {
  digitalWrite(GREEN_LED_PIN, LOW);
  digitalWrite(RED_LED_PIN, LOW);
  noTone(BUZZER_PIN);
}

void applyState(const String& state) {
  resetOutputs();

  if (state == "OK") {
    digitalWrite(GREEN_LED_PIN, HIGH);
  } else if (state == "OVERRIDE") {
    digitalWrite(RED_LED_PIN, HIGH);
  } else if (state == "CRITICAL") {
    digitalWrite(RED_LED_PIN, HIGH);
    tone(BUZZER_PIN, 2200, 300);
  }
}

void setup() {
  pinMode(GREEN_LED_PIN, OUTPUT);
  pinMode(RED_LED_PIN, OUTPUT);
  pinMode(BUZZER_PIN, OUTPUT);

  resetOutputs();
  Serial.begin(115200);
  while (!Serial) {
    delay(10);
  }
  Serial.println("IINTS UNO Q supervisor bridge ready");
  Serial.flush();
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      incomingLine.trim();
      if (incomingLine.length() > 0) {
        applyState(incomingLine);
        Serial.print("STATE=");
        Serial.println(incomingLine);
        Serial.flush();
      }
      incomingLine = "";
    } else {
      incomingLine += c;
    }
  }
}
