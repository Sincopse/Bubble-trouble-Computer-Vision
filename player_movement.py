import pygame


class PlayerMovement:
    @staticmethod
    def update_player_exact_position(player, position_relative):
        if position_relative != -1:
            player.xPosition = position_relative * 9

    @staticmethod
    def update_player_position(player):
        keys = pygame.key.get_pressed()
        if keys[player.left_key] == -1 and player.xPosition > player.velocity:
            player.xPosition -= player.velocity
        if keys[player.right_key] == 1 and player.xPosition < 900 - player.playerWidth - player.velocity:
            player.xPosition += player.velocity
