const int trigPin = 12; // Chân Trigger
const int echoPin = 13; // Chân Echo
long duration;
int distance;
const int servoPin = 3;
const int IN1 = 5;
const int IN2 = 4;
const int ENA = 6;
const int IN3 = 9;
const int IN4 = 8;
const int ENB = 10;

int rpi = 0;
void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  Serial.begin(9600);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(ENA, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  pinMode(ENB, OUTPUT);
}
void loop() {
  measureDistance();
  if (distance < 13) {
    // Xử lý tránh vật cản
    dung(100); 
  } 
   else{
    rpi = Serial.read() - 48; // Convert char to integer
    switch (rpi) {
      case 1: tien(14); break;
      case 3: retrai2(30); break;
      case 4: rephai2(35); break; 
      case 5: bussevor(); tien(14); delay(200);  break;
      case 8: rephai2(40); delay(4800); break;
      case 9: dung(500); break;
    }
  }
}
void measureDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  distance = duration * 0.034 / 2;
  delay(300);
}
void tien(int Speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, Speed);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, Speed);
}

void retrai2(int Speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, Speed);
}

void rephai2(int Speed) {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, Speed);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, 0);
}

void xoaytrai(int Speed) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, 35);
  digitalWrite(IN3, HIGH);
  digitalWrite(IN4, LOW);
  analogWrite(ENB, Speed);
}

void dung(int delayTime) {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  delay(delayTime);
}
void bussevor(){
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
  int pulseWidth1 = map(120, 0, 120, 544, 2400);
  digitalWrite(servoPin, HIGH);
  delayMicroseconds(pulseWidth1);
  digitalWrite(servoPin, LOW);
  delay(3000);
  int pulseWidth2 = map(90, 0, 120, 544, 2400);
  digitalWrite(servoPin, HIGH);
  delayMicroseconds(pulseWidth2);
  digitalWrite(servoPin, LOW);
  delay(200);
  
}

