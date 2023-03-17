from .no_source import DcNoSource


class DcSource(DcNoSource):
    def __init__(self, R:float, L:float, C:float, v0:float, i0:float, v_s:float, i_s:float) -> None:
        super().__init__(R, L, C, v0, i0)
        self.v_s = v_s
        self.i_s = i_s


class SeriesRLCSource(DcSource):
    ...


class ParallelRLCSource(DcSource):
    ...