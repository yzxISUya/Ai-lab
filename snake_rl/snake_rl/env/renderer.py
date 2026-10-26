"""Pygame 渲染器。

训练时**不要**创建它：窗口和帧率会严重拖慢采样。
它只服务于人类试玩和演示。
"""

from __future__ import annotations

import os

# 在 import pygame 之前设置，避免打印支持提示。
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from snake_rl.env.snake_env import Direction, Point, SnakeEnv, SnakeState

# 配色（深色主题）
BG = (18, 18, 24)
BOARD = (24, 24, 32)
GRID = (38, 38, 50)
SNAKE_BODY = (46, 204, 113)
SNAKE_HEAD = (120, 255, 170)
FOOD = (231, 76, 60)
TEXT = (236, 240, 241)
MUTED = (150, 158, 168)
BORDER = (60, 60, 78)


class PygameRenderer:
    """把 :class:`SnakeState` 画到窗口里。"""

    def __init__(
        self,
        env: SnakeEnv,
        cell_size: int = 24,
        caption: str = "Snake RL",
    ):
        self.env = env
        self.cell_size = cell_size
        self.board_width = env.config.width * cell_size
        self.board_height = env.config.height * cell_size
        self.hud_height = 62
        self.window_width = self.board_width
        self.window_height = self.board_height + self.hud_height

        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        font_size = 17 if self.window_width >= 420 else 14
        self.font = pygame.font.SysFont("consolas", font_size)
        if self.font is None:
            self.font = pygame.font.Font(None, font_size + 2)

    def draw(self, state: SnakeState, status: str = "") -> None:
        self.screen.fill(BG)
        pygame.draw.rect(
            self.screen,
            BOARD,
            pygame.Rect(0, 0, self.board_width, self.board_height),
        )

        self._draw_grid()
        self._draw_food(state.food)
        self._draw_snake(state)
        self._draw_border()
        self._draw_hud(state, status)
        pygame.display.flip()

    def tick(self, fps: int) -> None:
        self.clock.tick(fps)

    def should_quit(self) -> bool:
        """处理窗口关闭事件；训练/评估带渲染时用。"""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def close(self) -> None:
        pygame.display.quit()
        pygame.font.quit()

    # ------------------------------------------------------------------
    # 绘制细节
    # ------------------------------------------------------------------
    def _cell_rect(self, point: Point) -> pygame.Rect:
        size = self.cell_size
        return pygame.Rect(point.x * size, point.y * size, size, size)

    def _draw_grid(self) -> None:
        if self.cell_size < 10:
            return
        for x in range(self.env.config.width + 1):
            px = x * self.cell_size
            pygame.draw.line(self.screen, GRID, (px, 0), (px, self.board_height), 1)
        for y in range(self.env.config.height + 1):
            py = y * self.cell_size
            pygame.draw.line(self.screen, GRID, (0, py), (self.board_width, py), 1)

    def _draw_food(self, food: Point | None) -> None:
        if food is None:
            return
        rect = self._cell_rect(food).inflate(-self.cell_size * 0.45, -self.cell_size * 0.45)
        pygame.draw.ellipse(self.screen, FOOD, rect)

    def _draw_snake(self, state: SnakeState) -> None:
        size = self.cell_size
        # 从尾巴画到头，保证头在最上层。
        for point in reversed(state.snake[1:]):
            rect = self._cell_rect(point).inflate(-2, -2)
            pygame.draw.rect(self.screen, SNAKE_BODY, rect, border_radius=4)

        head_rect = self._cell_rect(state.head)
        pygame.draw.rect(self.screen, SNAKE_HEAD, head_rect, border_radius=5)
        self._draw_eyes(state.head, state.direction)

    def _draw_eyes(self, head: Point, direction: Direction) -> None:
        size = self.cell_size
        cx = head.x * size + size / 2
        cy = head.y * size + size / 2
        offset = max(2.0, size * 0.16)
        radius = max(1, int(size * 0.075))

        if direction == Direction.RIGHT:
            eyes = [(cx + offset, cy - offset), (cx + offset, cy + offset)]
        elif direction == Direction.LEFT:
            eyes = [(cx - offset, cy - offset), (cx - offset, cy + offset)]
        elif direction == Direction.UP:
            eyes = [(cx - offset, cy - offset), (cx + offset, cy - offset)]
        else:  # DOWN
            eyes = [(cx - offset, cy + offset), (cx + offset, cy + offset)]

        for ex, ey in eyes:
            pygame.draw.circle(self.screen, (20, 30, 25), (int(ex), int(ey)), radius)

    def _draw_border(self) -> None:
        pygame.draw.rect(
            self.screen,
            BORDER,
            pygame.Rect(0, 0, self.board_width, self.board_height),
            width=2,
        )

    def _draw_hud(self, state: SnakeState, status: str) -> None:
        stats = self.font.render(
            f"Score {state.score}  Steps {state.steps}  Len {len(state.snake)}",
            True,
            TEXT,
        )
        self.screen.blit(stats, (12, self.board_height + 9))

        if status:
            line = self.font.render(status, True, MUTED)
            self.screen.blit(line, (12, self.board_height + 34))
