import os
import pygame
from ple.games.flappybird import FlappyBird

def prepare_game():
    asset_dir = "../assets"

    game = FlappyBird()
    for c in game.images["player"]:
        image_assets = [
            os.path.join(asset_dir, "bird-upflap.png"),
            os.path.join(asset_dir, "bird-midflap.png"),
            os.path.join(asset_dir, "bird-downflap.png"),
        ]

        game.images["player"][c] = [pygame.image.load(im).convert_alpha() for im in image_assets]

    for b in game.images["background"]:
        game.images["background"][b] = pygame.image.load(os.path.join(asset_dir, "background.png")).convert()

    for c in ["red", "green"]:
        path = os.path.join(asset_dir, "pipe.png")

        game.images["pipes"][c] = {}
        game.images["pipes"][c]["lower"] = pygame.image.load(path).convert_alpha()
        game.images["pipes"][c]["upper"] = pygame.transform.rotate(game.images["pipes"][c]["lower"], 180)

    game.images["base"] = pygame.image.load(os.path.join(asset_dir, "base.png")).convert()

    return game
