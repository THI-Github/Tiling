import numpy as np
import random
from tkinter import *
from PIL import Image, ImageTk
from matplotlib import pyplot as plt

# TODO:
# - Gewichtete Wahrscheinlichkeiten an Pixelhelligkeit von eingelesenen Bildern ketten
# - Abbildung von Gesichtern etc. mit Tiles
# - Mehr als zwei Farben (0, 1, 2)


class TileSet:
    def __init__(self, side_logic, image_names, probs=None):
        self.tile_image_names = image_names
        self.tile_types_total = side_logic
        self.tile_types_names = np.array(range(len(side_logic)))

        if probs == None:
            # equal probabilty for all tiles
            self.tile_types_probs = {}
            for key in self.tile_types_names:
                self.tile_types_probs[key] = 1

        else:
            self.tile_types_probs = probs


class Grid:
    def __init__(self, max_row, max_col, tileset):
        self.max_row = max_row
        self.max_col = max_col
        self.tileset = tileset
        self.tile_image_names = tileset.tile_image_names
        self.grid = self.make_grid()

        #self.inherit_to_tiles()

    def make_grid(self):
        f = []
        for row in range(self.max_row):
            f_row = []
            for col in range(self.max_col):
                f_row.append(Tile(row, col, self))
            f.append(f_row)

        return np.array(f)

    def order_tiles(self):
        # return ordered list of grid-tiles ascending
        ordered_tiles = []
        for tile in np.ravel(self.grid):
            if self.get_possible_types(tile) != 0:
                ordered_tiles.append(tile)

        ordered_tiles = sorted(ordered_tiles, reverse=False, key=self.get_possible_types)
        return ordered_tiles

    def get_possible_types(self, tile):
        return len(tile.possible_types)

    def show_grid(self, output="test.png"):
        images = [Image.open(x) for x in self.tile_image_names]
        widths, heights = zip(*(i.size for i in images))

        total_width = max(widths) * self.max_col
        total_height = max(heights) * self.max_row

        new_im = Image.new('RGB', (total_width, total_height))

        for tile in np.ravel(self.grid):
            col = tile.col
            row = tile.row
            im = images[tile.tile_type]

            x = im.size[0] * col
            y = im.size[1] * row

            new_im.paste(im, (x, y))


        new_im.save(output)

    def compress_grid(self):
        # return compressed array with tile-types
        return np.array([[self.grid[row][col].tile_type for col in range(self.max_col)] for row in range(self.max_row)])

class Tile:
    dir_vector_mapping = np.array([
        [-1, 0],
        [0, 1],
        [1, 0],
        [0, -1]
    ])
    dir_side_mapping = np.array([
        2,
        3,
        0,
        1 # in direction (index) 0 the neighbor (in the north) sees the color in the South (->2)
    ])

    def __init__(self, row, col, parent_grid):
        self.row = row
        self.col = col
        self.grid_pos = np.array([row, col])
        self.given_sides = [-1, -1, -1, -1]
        self.tile_type = -1
        self.grid = parent_grid

        self.tile_types_total = parent_grid.tileset.tile_types_total
        self.tile_types_names = parent_grid.tileset.tile_types_names
        self.tile_types_probs = parent_grid.tileset.tile_types_probs
        self.possible_types = np.array(list(range(len(self.tile_types_total))))

    def set_type(self, typ):
        self.tile_type = typ
        self.possible_types = []
        self.given_sides = self.tile_types_total[self.tile_type]
        self.update_neighbors()

    def update_neighbors(self):
        # update neighbors
        for dir_idx in range(4):
            dir_pos = self.grid_pos + self.dir_vector_mapping[dir_idx]
            if dir_pos[0] in range(0, self.grid.max_row) and dir_pos[1] in range(0, self.grid.max_col):
                # neighbos is valid
                neighbor = self.grid.grid[dir_pos[0], dir_pos[1]]

                # check if neighbors tile_type must be decided
                if neighbor.tile_type == -1:
                    neighbor.given_sides[self.dir_side_mapping[dir_idx]] = self.given_sides[dir_idx]
                    neighbor.update_possible_types()

    def update_possible_types(self):
        allowed = self.possible_types
        filter_vertical = []
        for side_idx in range(len(self.given_sides)):
            side = self.given_sides[side_idx]

            if side == -1:
                out = np.array([1 for _ in range(len(self.tile_types_total))])
            else:
                out = [(side == self.tile_types_total[place, side_idx]) for place in range(len(self.tile_types_total))]


            filter_vertical.append(out)

        filter_vertical = np.array(filter_vertical)
        filter_true = [all(filter_vertical[:, i]) for i in range(len(self.tile_types_names))]
        self.possible_types = self.tile_types_names[filter_true]

    def choose_type(self):
        # randomly choose a type from the presented ones
        possible = list(self.possible_types)
        weights = [self.tile_types_probs[p] for p in possible]
        new_tile_type = random.choices(possible, weights, k=1)[0]
        self.set_type(new_tile_type)


def main():

    t1_probs = {
        0: 1,
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 0.1,
        7: 0.1
    }

    t2_probs = {
        0: 0.05,
        1: 2,
        2: 2,
        3: 0.25,
        4: 0.25,
        5: 0.25,
        6: 0.25,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 1
    }

    Tileset1 = TileSet(
        np.array([
        [0, 1, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]),
        [
            'tileset1/zero.png',
            'tileset1/one.png',
            'tileset1/two.png',
            'tileset1/three.png',
            'tileset1/four.png',
            'tileset1/five.png',
            'tileset1/six.png',
            'tileset1/seven.png'
        ],
        t1_probs
 )
    Tileset2 = TileSet(
        np.array([
            [1, 1, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0]
        ]),
        [
            'tileset_gang/zero.png',
            'tileset_gang/one.png',
            'tileset_gang/two.png',
            'tileset_gang/three.png',
            'tileset_gang/four.png',
            'tileset_gang/five.png',
            'tileset_gang/six.png',
            'tileset_gang/seven.png',
            'tileset_gang/eight.png',
            'tileset_gang/nine.png',
            'tileset_gang/ten.png',
            'tileset_gang/eleven.png'
        ],
        t2_probs
    )



    g1 = Grid(30, 30, Tileset2)

    #g2 = Grid('tileset_gang/local.png', Tileset2)

    step = 0
    while True:
        print(f"Step: {step} | Done: {step / (g1.max_row * g1.max_col) * 100} %")
        tile_list = g1.order_tiles()
        if len(tile_list) == 0:
             break
        else:
            tile_list[0].choose_type()
            step += 1

    #print("-"*20)
    #print(g1.compress_grid())
    g1.show_grid("test_3.png")


if __name__ == "__main__":
    main()





