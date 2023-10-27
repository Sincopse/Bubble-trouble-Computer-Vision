import pygame


class PlayerMovement:
    @staticmethod
    def update_player_position(player, cameraDirection):

        keys = pygame.key.get_pressed()
        if keys[player.left_key] or cameraDirection == -1 and player.xPosition > player.velocity:
            player.xPosition -= player.velocity
        if keys[player.right_key] or cameraDirection == 1 and player.xPosition < 900 - player.playerWidth - player.velocity:
            player.xPosition += player.velocity
