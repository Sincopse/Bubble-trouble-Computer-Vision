import pygame,math


class BubbleMovement():
	def __init__(self, position,screen,window_size,bubble_size):
		self.x, self.y = position
		self.width,self.height = window_size
		self.bubble_size = bubble_size
		self.screen = screen
		self.speed = 0
		self.angle = 0
		self.gravity = 0.1

	def display(self,image):
		self.screen.blit(image,(self.x,self.y))

	def move(self):
		self.x += math.sin(self.angle) * self.speed
		self.y -= math.cos(self.angle) * self.speed

	def bounce(self):
		self.y -= self.gravity
		if self.x > self.width - self.bubble_size:
			self.x = 2 * (self.width - self.bubble_size) - self.x
			self.angle = - self.angle
		elif self.x < 1:
			self.x = 2 *  self.x
			self.angle = - self.angle
		if self.y > self.height - self.bubble_size:
			self.y = 2 * (self.height - self.bubble_size) - self.y
			self.angle = math.pi - self.angle
		elif self.y < 1:
			self.y = 2 * self.y
			self.angle = math.pi - self.angle