# How to control RGB WS2812B  LED 32x8 matrix with an Arduino - Tutorial
- URL: https://www.youtube.com/watch?v=VtIWN5SJDYo
- Channel: Reid Projects | Smart Home & Tech
- Track via: api, en (uploader)

## Summary

**TL;DR:** Wire a 32×8 WS2812B matrix to an Arduino (common ground, data on pin 6, 5 V), install Adafruit NeoMatrix/NeoPixel/GFX, and edit the MatrixTest example for 32×8, top-left origin, row-major zigzag wiring to scroll text.

- [procedure] Wire matrix ground to Arduino ground, data-in to pin 6, and 5 V; full-white 256 LEDs may draw up to 15 A (no source), while text stayed under 2 A.
- [procedure] Install Adafruit NeoMatrix, NeoPixel, and GFX; open MatrixTest, set 32×8, top-left origin, row-major, zigzag wiring, then edit text and RGB intensities (0–255).
- [overview] Short hardware/screen demo; exact code and wiring diagram are on the linked project page, not fully in the transcript.

**Assessment:** Affiliate links to the matrix, Arduino, and power supply monetize the recommendations; the 15 A figure is unsourced. Check the linked code against current Adafruit library versions and verify power-supply safety before driving all LEDs at full brightness.

## Description

In this tutorial, you'll see how to display text on a 32x8 RGB #LED Matrix (built with #WS2812B) with an #Arduino within few minutes.

Please don't forget to Like 👍and Subscribe ✅ to get the next tutorials and projects !

If you need more details on the wiring or the code, have a look here : 
https://reid-projects.com/how-to-control-rgb-led-matrix-with-an-arduino/

*MATERIAL*

*32x8 RGB LED MATRIX*
➡️Amazon : https://amzn.to/38p2UXd
➡️AliExpress : https://s.click.aliexpress.com/e/_dUoVOQW

*ARDUINO UNO*
➡️Amazon  : https://amzn.to/2SkQf1U
➡️AliExpress : https://s.click.aliexpress.com/e/_dXABKCK

*5V POWER SUPPLY* (might requires up to 15amp)
➡️Amazon : https://amzn.to/38mCEMV
➡️AliExpress : https://s.click.aliexpress.com/e/_d7AidNo

However, you can use any matrix, including one built with LED strips.

If you built something with a similar setup, please comment and share your projects !

ℹ️ _As an Amazon Associate and SwitchBot affiliate, I earn from qualifying purchases. If you click on a link, I may receive a commission at no extra-cost for you. This helps me to keep creating and sharing content with you. Thanks !_
_For more details, see_ https://bit.ly/2Riiq0j

*FOLLOW US !*

Facebook : https://www.facebook.com/reidprojects/
Twitter : https://twitter.com/reidprojects
Instagram : https://www.instagram.com/reid_projects/
Youtube : https://www.youtube.com/c/reidprojects
Pinterest : https://www.pinterest.ca/reidprojects/

And check my other projects ! https://reid-projects.com

## Transcript

Hey everyone,  this is Alex Reid from Reid Projects and today we're going to see how to get started with the LED matrix and the Arduino.

First, I will use this one that I got from Amazon and you can find a link in the description below.

However, you can build it yourself with a LED strip or anything that has WS2812B.

So this one is a 32 by 8 so it has 256 leds and you can connect more than one together.

Here is the data in and that out, on our side we will of course use the data in and a first step will be to connect the ground of the LED matrix and the Arduino together then we'll do the same thing for the data.

For the data we'll use the pin and the pin 6 on the Arduino and finally the 5 volt.

note that this matrix 32 by 8 if we we turn on all the LEDs at the same time with the highest intensity you will need a 15 amp for the current so be sure that your power supply is enough if you need this.

On our side we will just display text and I didn't get more than 2 amps so that should be okay.

that's it ,so everything is wired now we just have to upload the code so far this will need three libraries from Adafruit the first one will be the NeoMatrix and this library will also requires two others, we'll have to download the Adafruit neopixel and Adafruit graphics and that's it.

So the Neomatrix lib comes with example so we just have to load the matrixtest.

This one shall be here should works out of the box.

That said, it is configured for a five by eight matrix with few particularities so we can just try to upload this code and see what happens  with the matrix that we have.

so we see that it's alive but it's not doing what we want so we'll add change some parameters of the example to fit with the new matrix we have so first is a 32 by 8 matrix not a 5 by 8 and also the first LED is on the top left not on the top right it's also configured by rows not by columns and the last thing is its wired is zigzag not in progressive so the the last lead of the row is connected with the last LED of the row below and the first with the first of the other row.

So it sends exact and then let's try to push this code great so now the example is running we can see text and it will try the three colors that are that are in the example now we can change you can try to change the text for anything you want you will also have to change some of the numbers just display if you have a longer text you will have to increase the numbers in the in dilute below so these are the three colors blue red green we can change the intensities to whatever we want so we can set it up to 255 and this is the tag that we will display.

Since I'm just changing the letters but keeping the same length, I don't have to play with the numbers.

And let's try to  you upload this and we see it's really brighter than it was first and it's running great so that's it,  this is the first quick start for the matrix with an Arduino but you can do whatever you want and this is as simple as that thanks for watching and don't forget to subscribe to see the next projects and tutorials
