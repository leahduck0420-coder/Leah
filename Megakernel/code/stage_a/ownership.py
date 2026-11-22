"""GF(2) program ownership of a tensor, with independent local coordinates."""

from dataclasses import dataclass
from enum import Enum
from itertools import product


class Mode(str, Enum):
    ROW = "row"
    COLUMN = "column"
    TILE_2D = "tile_2d"
    REPLICATE = "replicate"
    CUSTOM = "custom"


def bit_width(extent):
    return (extent - 1).bit_length()


def offsets(extents):
    offset = 0
    for extent in extents:
        yield offset
        offset += bit_width(extent)


def unpack(value, extents):
    return tuple((value >> offset) & ((1 << bit_width(extent)) - 1)
                 for extent, offset in zip(extents, offsets(extents)))


def pack(coords, extents):
    if len(coords) != len(extents) or any(not 0 <= c < d for c, d in zip(coords, extents)):
        raise ValueError("coordinate outside active domain")
    return sum(c << shift for c, shift in zip(coords, offsets(extents)))


def span(columns):
    values = [0]
    for column in columns:
        values.extend(value ^ column for value in values[:])
    return values


def apply(columns, point):
    value = 0
    while point:
        bit = point & -point
        value ^= columns[bit.bit_length() - 1]
        point ^= bit
    return value


class LinearSystem:
    def __init__(self, columns):
        self.pivots = {}
        self.kernel = []
        for bit, column in enumerate(columns):
            value, witness = self.reduce(column, 1 << bit)
            if value:
                self.pivots[value.bit_length()] = (value, witness)
            else:
                self.kernel.append(witness)

    def reduce(self, value, witness=0):
        for bit in sorted(self.pivots, reverse=True):
            if value & (1 << (bit - 1)):
                column, preimage = self.pivots[bit]
                value ^= column
                witness ^= preimage
        return value, witness


@dataclass(frozen=True)
class OwnershipMap:
    grid: tuple
    shape: tuple
    dtype: str
    columns: tuple
    local_mask: int

    def __post_init__(self):
        for field in ("grid", "shape", "columns"):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if not 1 <= len(self.grid) <= 3 or any(d < 1 for d in self.grid + self.shape):
            raise ValueError("invalid grid or tensor shape")
        if not self.dtype or self.program_bits > 63 or self.tensor_bits > 63:
            raise ValueError("empty dtype or unsupported packed width")
        if len(self.columns) != self.program_bits:
            raise ValueError("one column is required per program bit")
        limit = 1 << self.tensor_bits
        if not 0 <= self.local_mask < limit or any(
            not 0 <= column < limit or column & self.local_mask for column in self.columns
        ):
            raise ValueError("program columns must be disjoint from the local coordinate subspace")
        object.__setattr__(self, "_system", LinearSystem(self.columns))

    @property
    def program_bits(self):
        return sum(map(bit_width, self.grid))

    @property
    def tensor_bits(self):
        return sum(map(bit_width, self.shape))

    @property
    def nullity(self):
        return len(self._system.kernel)

    def copies(self):
        """Size of each nonempty fiber in the padded GF(2) domain."""
        return 1 << self.nullity

    def program_ids(self):
        return product(*(range(d) for d in self.grid))

    def tile(self, program):
        return apply(self.columns, pack(program, self.grid))

    def owners(self, element):
        tile = pack(element, self.shape) & ~self.local_mask
        residual, program = self._system.reduce(tile)
        if residual:
            return ()
        padded = (unpack(program ^ k, self.grid) for k in span(self._system.kernel))
        return tuple(sorted(p for p in padded if all(c < d for c, d in zip(p, self.grid))))

    def active_copies(self, element):
        return len(self.owners(element))

    def block_of(self, program):
        tile = self.tile(program)
        local_bits = [1 << bit for bit in range(self.tensor_bits) if self.local_mask & (1 << bit)]
        coords = (unpack(tile ^ point, self.shape) for point in span(local_bits))
        return tuple(sorted(e for e in coords if all(c < d for c, d in zip(e, self.shape))))

    def to_json(self):
        return {
            "grid": list(self.grid), "shape": list(self.shape), "dtype": self.dtype,
            "program_bits": self.program_bits, "columns": [hex(c) for c in self.columns],
            "local_mask": hex(self.local_mask), "copies": self.copies(),
        }

    @classmethod
    def from_json(cls, data):
        result = cls(data["grid"], data["shape"], data["dtype"],
                     [int(c, 16) for c in data["columns"]], int(data["local_mask"], 16))
        if result.copies() != data["copies"] or result.program_bits != data["program_bits"]:
            raise ValueError("inconsistent ownership metadata")
        return result


def from_grid(grid, shape, dtype, mode, *, columns=None, local_mask=None):
    grid, shape = tuple(grid), tuple(shape)
    mode = Mode(mode)
    if mode is Mode.CUSTOM:
        if columns is None or local_mask is None:
            raise ValueError("custom ownership requires columns and a local mask")
        return OwnershipMap(grid, shape, dtype, columns, local_mask)
    if mode is Mode.TILE_2D and (len(grid) < 2 or len(shape) < 2):
        raise ValueError("2-D tiling requires two grid and tensor dimensions")
    axes = {
        Mode.ROW: {0: 0},
        Mode.COLUMN: {0: len(shape) - 1},
        Mode.TILE_2D: {0: 0, 1: 1},
        Mode.REPLICATE: {},
    }[mode]
    tensor_width = sum(map(bit_width, shape))
    columns = [0] * sum(map(bit_width, grid))
    local_mask = (1 << tensor_width) - 1
    grid_offsets, tensor_offsets = list(offsets(grid)), list(offsets(shape))
    for grid_axis, tensor_axis in axes.items():
        if not 0 <= grid_axis < len(grid) or not 0 <= tensor_axis < len(shape):
            raise ValueError("invalid axis assignment")
        program_width, coord_width = bit_width(grid[grid_axis]), bit_width(shape[tensor_axis])
        if program_width > coord_width:
            raise ValueError("grid overpartitions the tensor coordinate bits")
        start = tensor_offsets[tensor_axis] + coord_width - program_width
        for bit in range(program_width):
            column = 1 << (start + bit)
            columns[grid_offsets[grid_axis] + bit] = column
            local_mask ^= column
    return OwnershipMap(grid, shape, dtype, columns, local_mask)
