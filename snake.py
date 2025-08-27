from neural_network import *


class Snake:
    def __init__(self, neural_net=None):
        self.body = [[10, 10], [9, 10], [9, 11], [9, 12]]      
        self.head = self.body[0][:]                             
        self.old_tail = self.head[:]                            
        self.direction = RIGHT
        self.age = 0
        self.starve = 500                                      
        self.alive = True
        self.neural_net = neural_net
        self.vision = []                                        

    def update(self):
        self.age += 1
        self.starve -= 1
        if self.starve < 1:
            self.alive = False
        self.move()

    def grow(self):
        self.starve = 500                   
        self.body.append(self.old_tail)     

    def move(self):
        self.old_tail = self.body[-1][:]       
        self.head[0] += self.direction[0]     
        self.head[1] += self.direction[1]
        if self.head in self.body[1:]:        
            self.alive = False
        self.body.insert(0, self.body.pop())   
        self.body[0] = self.head[:]          

    def turn_right(self):
        temp = self.direction[0]
        self.direction[0] = -self.direction[1]
        self.direction[1] = temp

    def turn_left(self):
        temp = self.direction[0]
        self.direction[0] = self.direction[1]
        self.direction[1] = -temp

    def AI(self):
        decision = np.argmax(self.neural_net.feed_forward(self.vision))
        if decision == 1:
            self.turn_right()
        elif decision == 2:
            self.turn_left()

    def fitness(self):
        return (len(self.body)**2) * self.age

    def render(self, window):
        body = pygame.image.load(IMAGE_SNAKE).convert_alpha()                 
        for block in self.body:
            window.blit(body, (block[0]*SPRITE_SIZE, block[1]*SPRITE_SIZE))    
        if self.neural_net:                                                     
            self.neural_net.render(window, self.vision)
