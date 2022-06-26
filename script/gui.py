from os import environ
environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

from os.path import join, dirname
import numpy as np
import pygame as pg
from sys import exit
from time import sleep

from iothello.backend_gui import Othello


class Gui(Othello):

    def __init__(self):
        super().__init__()
        pg.init()
        self.clock = pg.time.Clock()
        self.FR = 60
        self.clock.tick(self.FR)
        self.ROWS, self.COLS = 6, 6
        self.SQUARE_SIZE = 100
        self.SCREEN_SIZE = (600, 600)
        self.RED = (255, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GREEN1 = (0, 150, 0)
        self.GREEN2 = (0, 100, 0)
        self.screen = pg.display.set_mode(self.SCREEN_SIZE)
        pg.display.set_caption('OTHELLO')
        self.font = join(dirname(__file__), 'font.otf')
        self.end_game = False
        self.score = None

    def main_menu(self):
        selection_is_up = True
        while True:
            self.update_main_menu(selection_is_up)
            event = pg.event.wait()
            if event.type == pg.QUIT:
                pg.quit()
                exit()
            elif event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    selection_is_up = True
                elif event.key == pg.K_DOWN:
                    selection_is_up = False
                elif event.key == pg.K_RETURN:
                    if selection_is_up:
                        self.play()
                    else:
                        pg.quit()
                        exit()

    def update_main_menu(self, selection_is_up):
        self.screen.fill(self.GREEN1)

        if not self.end_game:
            title = self.text_format("OTHELLO", 60, self.RED)
        else:
            title = self.text_format("END", 60, self.RED)
            if self.score < 0:
                score = self.text_format('BLACK WINS', 60, self.BLACK)
            elif self.score == 0:
                score = self.text_format("IT'S A TIE", 60, self.WHITE)
            else:
                score = self.text_format('WHITE WINS', 60, self.WHITE)
            self.screen.blit(score, (self.SCREEN_SIZE[0] / 2 - (score.get_rect()[2] / 2), 200))

        if selection_is_up:
            text_start = self.text_format("PLAY", 40, self.WHITE)
            text_quit = self.text_format("QUIT", 40, self.BLACK)
        else:
            text_start = self.text_format("PLAY", 40, self.BLACK)
            text_quit = self.text_format("QUIT", 40, self.WHITE)

        self.screen.blit(title, (self.SCREEN_SIZE[0] / 2 - (title.get_rect()[2] / 2), 100))
        self.screen.blit(text_start, (self.SCREEN_SIZE[0] / 2 - (text_start.get_rect()[2] / 2), 300))
        self.screen.blit(text_quit, (self.SCREEN_SIZE[0] / 2 - (text_quit.get_rect()[2] / 2), 360))
        pg.display.update()
        pg.display.set_caption("OTHELLO")

    def draw_squares(self):
        self.screen.fill(self.GREEN1)
        for row in range(self.ROWS):
            for col in range(row % 2, self.COLS, 2):
                pg.draw.rect(self.screen, self.GREEN2, (row * self.SQUARE_SIZE, col * self.SQUARE_SIZE, self.SQUARE_SIZE, self.SQUARE_SIZE))

    def find_player_color(self, player):
        if player == 1:
            return self.WHITE
        else:
            return self.BLACK

    def draw_piece(self):
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if self.board[row, col] != 0:
                    pg.draw.circle(self.screen, self.find_player_color(self.board[row][col]),
                                   (int(col * self.SQUARE_SIZE + self.SQUARE_SIZE // 2), int(row * self.SQUARE_SIZE + self.SQUARE_SIZE // 2)),
                                   int(self.SQUARE_SIZE * 0.4))

    def text_format(self, message, textSize, textColor):
        newFont = pg.font.Font(self.font, textSize)
        newText = newFont.render(message, 0, textColor)
        return newText

    def play(self):
        player = 1
        move_count = 0
        self.draw_squares()
        self.draw_piece()

        while True:
            pg.display.update()

            moves = self.find_possible_moves(player)
            if len(moves) == 0:
                player = -player
                moves = self.find_possible_moves(player)
                if len(moves) == 0:
                    pg.time.delay(3000)
                    self.score = np.sum(self.board)
                    self.reset_board()
                    self.end_game = True
                    self.main_menu()

            if player == 1:
                while True:
                    event = pg.event.wait()
                    if event.type == pg.QUIT:
                        pg.quit()
                        exit()
                    elif event.type == pg.MOUSEBUTTONDOWN:
                        mousex, mousey = event.pos
                        move = (mousey // self.SQUARE_SIZE, mousex // self.SQUARE_SIZE)
                        if move in moves:
                            self.update_board(move, 1)
                            break

            elif player == -1:
                sleep(0.3)
                if move_count == 31:
                    move = self.find_best_move_simple()
                else:
                    move = self.find_best_move_model(move_count)
                self.update_board(move, -1)

            move_count += 1
            self.draw_piece()
            player = -player


def play():
    Gui().main_menu()