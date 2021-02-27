# include <Stepper.h>
int stepsPerRevolution = 2048;
Stepper myStepper(stepsPerRevolution,8,10,9,11);
int motSpeed=10;
int delayTime=500;
int direct;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  myStepper.setSpeed(motSpeed);
}

void loop() {
  // put your main code here, to run repeatedly:
  if (Serial.available()>0){
    direct = Serial.parseInt();

    if (direct == 1){
      Serial.println("you will go to clockwise");
      myStepper.step(stepsPerRevolution/9);
    }

    if (direct == 2){
      Serial.println("you will go to counterclockwise");
      myStepper.step(-stepsPerRevolution/9);
    }
  
  } 
}
