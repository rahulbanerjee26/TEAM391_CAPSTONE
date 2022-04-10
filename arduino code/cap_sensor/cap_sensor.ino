 #include <CapacitiveSensor.h>
CapacitiveSensor Sensor = CapacitiveSensor(4, 6);
long val;
int pos;
#define led 13

void setup()
{
Serial.begin(9600);
pinMode(led, OUTPUT);
}

void loop()
{

val = Sensor.capacitiveSensor(30);
Serial.println(val);

}
