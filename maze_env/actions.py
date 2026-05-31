"""动作枚举与运动向量映射。

所有需要动作编码的模块统一从此处导入，避免魔法数字散落各处。

Example:
    >>> from maze_env.actions import Action
    >>> env.step(Action.RIGHT)
    >>> Action.UP.value
    0
"""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """四方向离散动作枚举。

    值与 ``Discrete(4)`` 动作空间编号一一对应：

    +-------+-------+-----------+
    | 枚举  | 值    | 方向       |
    +=======+=======+===========+
    | UP    | 0     | Δrow = -1  |
    +-------+-------+-----------+
    | DOWN  | 1     | Δrow = +1  |
    +-------+-------+-----------+
    | LEFT  | 2     | Δcol = -1  |
    +-------+-------+-----------+
    | RIGHT | 3     | Δcol = +1  |
    +-------+-------+-----------+
    """

    UP    = 0
    DOWN  = 1
    LEFT  = 2
    RIGHT = 3


# (Δrow, Δcol) 与 Action 值的对应表，下标即 Action 枚举值
DELTAS: tuple[tuple[int, int], ...] = (
    (-1,  0),   # Action.UP
    ( 1,  0),   # Action.DOWN
    ( 0, -1),   # Action.LEFT
    ( 0,  1),   # Action.RIGHT
)
