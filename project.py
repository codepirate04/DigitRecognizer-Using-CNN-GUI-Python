import sys, pygame 
from pygame.locals import *
from trainedmodel import classify
import time
pygame.init()

screen= pygame.display.set_mode((300,300))
pygame.display.set_caption('Digit recognition @CodePirate')

brush= pygame.image.load('brushwhite.png')
brush =pygame.transform.scale(brush,(32,32))
pygame.display.update()


z=0

while True:
    x,y=pygame.mouse.get_pos()
    for event in pygame.event.get():
        if(event.type==pygame.QUIT):
            sys.exit()
        elif(event.type==MOUSEBUTTONDOWN):
            z=1
        elif(event.type==MOUSEBUTTONUP):
            z=0
        elif(event.type==pygame.KEYDOWN):
            if(event.key==K_SPACE):
                pygame.image.save(screen,"image.png")
                time.sleep(2)
                classify()
            elif(event.key==K_BACKSPACE):
                screen.fill((0,0,0))
                pygame.display.update()
        if(z==1):
            screen.blit(brush,(x-16,y-16))
            pygame.display.update()