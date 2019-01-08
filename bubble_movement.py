import pygame
import math

BUBBLE_SIZER = 20
AMPLITUDE_SIZER = 2

class BubbleMovement():
	def __init__(self, position,screen,window_size,bubble_size,amplitude,collisionTime,img):
		self.x, self.y = position
		self.width,self.height = window_size
		self.bubble_size = bubble_size
		self.screen = screen
		self.speed = 0
		self.angle = 0
		self.firstTouch = True
		self.amplitude = amplitude
		self.hitbox = (self.x, self.y, self.bubble_size, self.bubble_size)
		self.img = img
		self.collisionTime = collisionTime

	def display(self,image):
		self.screen.blit(image,(self.x,self.y))

	def move(self):
		self.x += (math.sin(self.angle) * self.speed) / self.amplitude
		self.y -= (math.cos(self.angle) * self.speed)


	def bounce(self):
		if self.x > self.width - self.bubble_size:
			self.x = 2 * (self.width - self.bubble_size) - self.x
			self.angle = - self.angle
		elif self.x < 1:
			if self.x < 0:
				self.x = 2 * (-self.x)
			else:
				self.x = 2 * self.x
			self.angle = - self.angle

		if self.firstTouch == False and self.y <= self.height / 3 :
			self.y = 3 * (self.height / 3 - self.bubble_size) - self.y
			self.angle = math.pi - self.angle

		if self.y > self.height - self.bubble_size:
			self.y = 2 * (self.height - self.bubble_size) - self.y
			self.angle = math.pi - self.angle
			self.firstTouch = False
		elif self.y < 1:
			self.y = 2 * self.y
			self.angle = math.pi - self.angle

	def collision(self,projectile1,projectile2):
		#PLAYER 1
		if self.y + self.hitbox[2] > projectile1.hitbox[1] and self.y < projectile1.hitbox[1] + projectile1.hitbox[3]:
			if self.x + self.hitbox[2] > projectile1.hitbox[0] and self.x < projectile1.hitbox[0] + projectile1.hitbox[2]:
				self.collisionTime += 1
				self.bubble_size -= BUBBLE_SIZER * self.collisionTime
				self.amplitude -= AMPLITUDE_SIZER * self.collisionTime
				return (True,1)

		#PLAYER 2
		if self.y + self.hitbox[2] > projectile2.hitbox[1] and self.y < projectile2.hitbox[1] + projectile2.hitbox[3]:
			if self.x + self.hitbox[2] > projectile2.hitbox[0] and self.x < projectile2.hitbox[0] + projectile2.hitbox[2]:
				self.collisionTime += 1
				self.bubble_size -= BUBBLE_SIZER * self.collisionTime
				self.amplitude -= AMPLITUDE_SIZER * self.collisionTime
				return (True,2);

		return (False,1)