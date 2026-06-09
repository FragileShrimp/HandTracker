import pyautogui
import keyboard
import time
   

def recurssion():
   count = 0  
   while True:
      print(count)
      count+=1
      time.sleep(.5)
      
      if keyboard.is_pressed("q"):
         break
      elif keyboard.is_pressed("e"):
         exit()
   
   recurssion()

def main():
   recurssion()

main()