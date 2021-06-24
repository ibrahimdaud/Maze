import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pygame.locals import *
import pygame


# Required constants, values of each element in matrix and reward for each
EMPTY_SPACE = 0
EMPTY_SPACE_COST = 0.05
WALL = 1
WALL_PENALTY = 0.75
MILESTONE = 3
MILESTONE_REWARD = 0.25
SLOW_PATH = 2
SLOW_PATH_COST = 0.2

STARTING_POSITION = 9

ENDING_POSITION = 8
ENDING_REWARD = 1

OUT_OF_BOUNDS_PENALTY = 0.75

CMAP = ListedColormap(['k', 'r', 'pink', 'blue', 'k', 'k', 'k', 'k', 'green', 'yellow'])
pygameColours = [
    (0, 0, 0),
    (255, 0, 0),
    (255, 100, 100),
    (0, 0, 255),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 255, 0),
    (255, 255, 0)
]


def state2ij(state):
    """
    Takes as input a state (int) and returns the x, y coordinate pair it 
    represents
    """
    i = int(state/8)
    j = state - i*8
    return i, j


def ij2state(x, y):
    """
    Takes as inputs the x, y coordinate pair of a space in the maze and returns
    its state (int)
    """
    return 8*x + y


def addArrow(pos, direction):
    """
    Draws an arrow in the current pyplot plot in the position "pos" and
    looking at direction "direction"

    pos - (x, y) coordinate pair, int tuple
    direction - (dx, dy) direction coordinate pair, {-1, 0, 1} tuple
    """

    if direction == (1, 0):
        # right arrow
        plt.arrow(pos[0] - 0.4, pos[1], 0.4, 0, head_width=0.3, color="white")
    elif direction == (-1, 0):
        # left arrow
        plt.arrow(pos[0] + 0.4, pos[1], -0.4, 0, head_width=0.3, color="white")
    elif direction == (0, -1):
        # top arrow
        plt.arrow(pos[0], pos[1] + 0.4, 0, -0.4, head_width=0.3, color="white")
    elif direction == (0, 1):
        # down arrow
        plt.arrow(pos[0], pos[1] - 0.4, 0, 0.4, head_width=0.3, color="white")


def path2arrows(path):
    """
    Draws arrows in the given input path in the current pyplot plot
    """
    previousState = path[0]
    for state in path[1:]:
        dx = state[0] - previousState[0]
        dy = state[1] - previousState[1]
        addArrow((previousState[0], previousState[1]), (dx, dy))
        previousState = state


class MazeSolver:
    def __init__(self, maze):
        """
        Class that contains the logic and useful functions for the agent
        playing the maze.

        Arguments:
        maze - nxm matrix of integers representing the different elements:
                0 - empty space
                1 - wall
                2 - slow path
                3 - milestone
                9 - starting position (should have only one)
                8 - ending position

        Properties:
        originalMaze - The maze the agent has to solve, given as input.
        Qtable - Table containing the learned score value for each action in
                 each state.

        Methods:
        getInitialPosition
        availableActions
        playMaze
        trainMaze
        drawSolution
        drawQtable
        """
        self.originalMaze = maze
        mazeShape = maze.shape
        numberOfStates = mazeShape[0]*mazeShape[1]
        self.Qtable = np.zeros(shape=[numberOfStates, 4]) + 0.0000001
        # self.Qtable = np.random.uniform(low=-1, size=[numberOfStates, 4])

    def getInitialPosition(self):
        """
        Searches the maze for the starting position element
        """
        for i in range(self.originalMaze.shape[0]):
            for j in range(self.originalMaze.shape[1]):
                if self.originalMaze[i, j] == STARTING_POSITION:
                    return (j, i)

    def availableActions(self, currentPosition, visited):
        """
        Returns a list of available actions given the current position and the
        list of visited positions.
        """
        available = []
        for action in range(4):
            dx = 0
            dy = 0
            if action == 0:
                dx = -1
            elif action == 1:
                dx = 1
            elif action == 2:
                dy = -1
            elif action == 3:
                dy = 1

            validMove = True

            attemptedPosition = (currentPosition[0] + dy,
                                 currentPosition[1] + dx)

            a, b = attemptedPosition

            height, width = self.originalMaze.shape
            inXRange = a < width and a >= 0
            inYRange = b < height and b >= 0
            if not inXRange or not inYRange:
                validMove = False
            elif self.originalMaze[b, a] == WALL:
                validMove = False
            elif attemptedPosition in visited:
                validMove = False

            if validMove:
                available.append(action)

        return available

    def playMaze(self, discover=False, update=True, lr=0.01, gamma=0.9):
        """
        Makes the agent play the maze one time and returns the path it took
        and the score it got.

        Optional arguments:
        discover - Boolean (default False). If False the agent will take the
                   path it thinks is the best, if True the agent will try
                   different paths some times.

        update - Boolean (default True). If True the agent will use the rewards
                 it gets to learn a better path, if false it will not learn
                 from playing.

        lr - Float (default 0.01), the learning rate of the agent
        gamma - Float (default 0.9), the parameter gamma of the update function
        """
        ended = False
        initialPosition = self.getInitialPosition()
        currentPosition = initialPosition
        path = [currentPosition]
        visited = set([currentPosition])
        Qtable = self.Qtable
        height, width = self.originalMaze.shape
        score = 0
        iterations = 0
        while not ended:
            iterations += 1
            # Selects an action
            if discover:
                # probs = np.array(Qtable[ij2state(*currentPosition)])
                # probs = np.exp(probs)/np.sum(np.exp(probs))
                if np.random.uniform() < 0.05:
                    probs = None
                    action = np.random.choice([0, 1, 2, 3], p=probs)
                else:
                    action = np.argmax(Qtable[ij2state(*currentPosition)])

            else:
                action = np.argmax(Qtable[ij2state(*currentPosition)])

            # Determines what the action means
            dx = 0
            dy = 0
            if action == 0:
                dx = -1
            elif action == 1:
                dx = 1
            elif action == 2:
                dy = -1
            elif action == 3:
                dy = 1

            attemptedPosition = (currentPosition[0] + dy,
                                 currentPosition[1] + dx)

            # Decides if the move is valid, if it ends the game and the reward
            # it yields
            validMove = True
            ended = False
            a, b = attemptedPosition

            inXRange = a < width and a >= 0
            inYRange = b < height and b >= 0
            if not inXRange or not inYRange:
                reward = -OUT_OF_BOUNDS_PENALTY
                validMove = False
            elif self.originalMaze[b, a] == WALL:
                reward = -WALL_PENALTY
                validMove = False
            elif self.originalMaze[b, a] == MILESTONE:
                reward = MILESTONE_REWARD
            elif self.originalMaze[b, a] == SLOW_PATH:
                reward = -SLOW_PATH_COST
            elif self.originalMaze[b, a] == EMPTY_SPACE:
                reward = -EMPTY_SPACE_COST
            elif self.originalMaze[b, a] == ENDING_POSITION:
                reward = ENDING_REWARD
                ended = True
                score += 100

            # if attemptedPosition in visited:
            #     validMove = False
            #     if len(self.availableActions(currentPosition, visited)) == 0:
            #         ended = True
            #         score = 0
            #         reward = -OUT_OF_BOUNDS_PENALTY
            #     elif not discover:
            #         ended = True

            # Updates Qtable if required, adds the move to the path if it is valid
            if validMove:
                if update:
                    Qtable[ij2state(*currentPosition), action] += lr * (reward + gamma * np.max(Qtable[ij2state(*attemptedPosition), :]) - Qtable[ij2state(*currentPosition), action])
                currentPosition = attemptedPosition
                score += reward
                path.append(currentPosition)
                visited.add(currentPosition)
            elif update:
                Qtable[ij2state(*currentPosition), action] += reward*lr

            if not validMove and not discover:
                ended = True
                path.append(attemptedPosition)

        return path, score

    def trainAgent(self, epochs=5000, verbose=False, lr=0.1, gamma=0.9, fps=50):
        """
        Trains the agent for the given amount of epochs

        Optional Arguments:
        epochs - Integer (default 5000), number of times the agent will attempt
                 to solve the maze to learn the best actions it can make.
        verbose - Boolean (defaul False), if True the function will print in
                  console the results after each epoch
        lr - Float (default 0.01), the learning rate of the agent
        gamma - Float (default 0.9), the parameter gamma of the update function
        """
        pygame.init()
        screen = pygame.display.set_mode((400, 400))
        done = False
        # if showEvery is None:
        #     showEvery = epochs + 1
        # else:
        #     print("Draw initial solution")
        #     self.drawSolution()
        for epoch in range(epochs):
            path, score = self.playMaze(discover=True,
                                        update=True,
                                        lr=lr,
                                        gamma=gamma)
            if verbose:
                if score > 0:
                    print(f"Found a solution with {len(path)} steps with score {score}")
                else:
                    print("The agent got trapped")
            self.drawPath(screen, path, fps)
            pygame.draw.rect(screen, (0, 0, 0), (0, 0, 400, 400))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
            if done:
                break
            pygame.display.flip()
            pygame.time.delay(1)

    def drawMaze(self, surface):
        rectWidth = 400/self.originalMaze.shape[0]
        rectHeight = 400/self.originalMaze.shape[1]
        for i in range(self.originalMaze.shape[0]):
            for j in range(self.originalMaze.shape[1]):
                rect = pygame.Rect(j*50, i*50, 50, 50)
                pygame.draw.rect(surface, pygameColours[self.originalMaze[i, j]], rect)

    def drawPath(self, surface, path, fps):
        for i, j in path:
            self.drawMaze(surface)
            rect = pygame.Rect(i*50+15, j*50+15, 20, 20)
            pygame.draw.ellipse(surface, (255, 255, 255), rect)
            pygame.display.flip()
            pygame.time.delay(int(1000/fps))

    def drawSolution(self):
        """
        Draws the matrix in a pyplot plot and draws arrows showing the path
        the agent takes to solve the maze
        """
        path, score = self.playMaze(update=False)
        plt.matshow(maze, cmap=CMAP)
        path2arrows(path)
        plt.draw()

    def drawQtable(self):
        """
        Draws the matrix in a pyplot plot and draws the max score of the
        actions that can be taken from each space.
        """
        plt.matshow(maze, cmap=CMAP)

        for i in range(maze.shape[0]):
            for j in range(maze.shape[1]):
                value = np.max(self.Qtable[ij2state(j, i), :])
                plt.text(j - 0.5, i, str(int(value*100)/100))

        plt.show()


if __name__ == "__main__":
    maze = np.array([
        [8, 0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 1, 0, 0],
        [1, 0, 1, 9, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]
    ])
    solver = MazeSolver(maze)
    solver.trainAgent(verbose=True, epochs=200, lr=0.01, fps=50)
    solver.drawSolution()
    plt.show()
