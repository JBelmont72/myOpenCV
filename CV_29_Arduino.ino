/*
 lesson we show you how you can control objects in the real world using OpenCV, Python, Mediapipa and our old friend, the Arduino. On the Python side, we recognize hand gestures, and then we pass the recognized gesture to Arduino and Arduino lights LED in response to what hand signal is seen. This is a simple example, but a very powerful method. Instead of LED, you could operate servos, stepper motors or relays to control any manner of different devices. For your convenience, this is the code we used on the Arduino side


*/
String cmd;
int LED1=9;
int LED2=6;
int LED3=5;
int LED4=3;
int LED5=10;
void setup() {
  pinMode(LED1,OUTPUT);
  pinMode(LED2,OUTPUT);
  pinMode(LED3,OUTPUT);
   pinMode(LED4,OUTPUT);
   pinMode(LED5,OUTPUT);
 Serial.begin(115200);
 
}
void loop() {
 while (Serial.available()==0){
 
 }
 cmd = Serial.readStringUntil('\r');
if (cmd=="One"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
}
if (cmd=="Two"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
}
 if (cmd=="Three"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,HIGH);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
 }
 if (cmd=="Four"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,HIGH);
  digitalWrite(LED4,HIGH);
  digitalWrite(LED5,LOW);
 }
 if (cmd=="Five"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,HIGH);
  digitalWrite(LED4,HIGH);
  digitalWrite(LED5,HIGH);
 
 }
 
 if (cmd=="Pinky"){
  digitalWrite(LED1,LOW);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,HIGH);
  digitalWrite(LED5,LOW);
 
 }
  if (cmd=="Thumb"){
  digitalWrite(LED1,LOW);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,HIGH);
 
 }
  if (cmd=="Inside"){
  digitalWrite(LED1,LOW);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,HIGH);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
 
 }
   if (cmd=="Outside"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,HIGH);
  digitalWrite(LED5,LOW);
 
 }
    if (cmd=="Unknown"){
  digitalWrite(LED1,LOW);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
 
 }
}'''
 lesson we show you how you can control objects in the real world using OpenCV, Python, Mediapipa and our old friend, the Arduino. On the Python side, we recognize hand gestures, and then we pass the recognized gesture to Arduino and Arduino lights LED in response to what hand signal is seen. This is a simple example, but a very powerful method. Instead of LED, you could operate servos, stepper motors or relays to control any manner of different devices. For your convenience, this is the code we used on the Arduino side


'''
String cmd;
int LED1=9;
int LED2=6;
int LED3=5;
int LED4=3;
int LED5=10;
void setup() {
  pinMode(LED1,OUTPUT);
  pinMode(LED2,OUTPUT);
  pinMode(LED3,OUTPUT);
   pinMode(LED4,OUTPUT);
   pinMode(LED5,OUTPUT);
 Serial.begin(115200);
 
}
void loop() {
 while (Serial.available()==0){
 
 }
 cmd = Serial.readStringUntil('\r');
if (cmd=="One"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
}
if (cmd=="Two"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
}
 if (cmd=="Three"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,HIGH);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
 }
 if (cmd=="Four"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,HIGH);
  digitalWrite(LED4,HIGH);
  digitalWrite(LED5,LOW);
 }
 if (cmd=="Five"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,HIGH);
  digitalWrite(LED4,HIGH);
  digitalWrite(LED5,HIGH);
 
 }
 
 if (cmd=="Pinky"){
  digitalWrite(LED1,LOW);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,HIGH);
  digitalWrite(LED5,LOW);
 
 }
  if (cmd=="Thumb"){
  digitalWrite(LED1,LOW);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,HIGH);
 
 }
  if (cmd=="Inside"){
  digitalWrite(LED1,LOW);
  digitalWrite(LED2,HIGH);
  digitalWrite(LED3,HIGH);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
 
 }
   if (cmd=="Outside"){
  digitalWrite(LED1,HIGH);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,HIGH);
  digitalWrite(LED5,LOW);
 
 }
    if (cmd=="Unknown"){
  digitalWrite(LED1,LOW);
  digitalWrite(LED2,LOW);
  digitalWrite(LED3,LOW);
  digitalWrite(LED4,LOW);
  digitalWrite(LED5,LOW);
 
 }
}