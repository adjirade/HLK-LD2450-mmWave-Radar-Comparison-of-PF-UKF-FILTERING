// ── Pin & konfigurasi ──────────────────────────────────────
#define RADAR_RX_PIN   16       // ESP32 RX2 ← LD2450 TX
#define RADAR_TX_PIN   17       // ESP32 TX2 → LD2450 RX
#define RADAR_BAUD     256000   // Baud rate sensor (default pabrik)
#define PC_BAUD        115200   // Baud rate ke laptop
#define FRAME_SIZE     30       // Panjang frame tetap

HardwareSerial radarSerial(2);


enum ParseState { WAIT_H1, WAIT_H2, WAIT_H3, WAIT_H4, READ_DATA };
ParseState state = WAIT_H1;

uint8_t  frame[FRAME_SIZE];
uint8_t  frameIdx = 0;

// ── Konversi koordinat ─────────────────────────────────────
int16_t decodeCoord(uint8_t lo, uint8_t hi) {
  uint16_t raw = (uint16_t)lo | ((uint16_t)hi << 8);
  int16_t  val = (int16_t)(raw & 0x7FFF);
  return (raw & 0x8000) ? val : -val;
}

// ── Proses & kirim satu frame ke laptop ───────────────────
void processFrame() {
  // Validasi
  if (frame[28] != 0x55 || frame[29] != 0xCC) return;

  int16_t x[3], y[3];
  for (int i = 0; i < 3; i++) {
    int o  = 4 + (i * 8);         // offset: lewati header 4 byte
    x[i]   = decodeCoord(frame[o + 0], frame[o + 1]);
    y[i]   = decodeCoord(frame[o + 2], frame[o + 3]);
  }

  // Kirim ke laptop via USB Serial
  Serial.print(x[0]); Serial.print(",");
  Serial.print(y[0]); Serial.print(",");
  Serial.print(x[1]); Serial.print(",");
  Serial.print(y[1]); Serial.print(",");
  Serial.print(x[2]); Serial.print(",");
  Serial.println(y[2]);
}

// ── State machine: cari header byte per byte ───────────────
void parseByte(uint8_t b) {
  switch (state) {
    case WAIT_H1: if (b == 0xAA) { frame[0] = b; state = WAIT_H2; } break;
    case WAIT_H2: if (b == 0xFF) { frame[1] = b; state = WAIT_H3; } else state = WAIT_H1; break;
    case WAIT_H3: if (b == 0x03) { frame[2] = b; state = WAIT_H4; } else state = WAIT_H1; break;
    case WAIT_H4:
      if (b == 0x00) { frame[3] = b; frameIdx = 4; state = READ_DATA; }
      else state = WAIT_H1;
      break;
    case READ_DATA:
      frame[frameIdx++] = b;
      if (frameIdx >= FRAME_SIZE) {
        processFrame();
        state    = WAIT_H1;
        frameIdx = 0;
      }
      break;
  }
}

// ── Setup ──────────────────────────────────────────────────
void setup() {
  Serial.begin(PC_BAUD);          // USB Serial → laptop (115200)
  radarSerial.begin(RADAR_BAUD, SERIAL_8N1, RADAR_RX_PIN, RADAR_TX_PIN); // UART2
}

// ── Loop ───────────────────────────────────────────────────
void loop() {
  while (radarSerial.available()) {
    parseByte(radarSerial.read());
  }
}
