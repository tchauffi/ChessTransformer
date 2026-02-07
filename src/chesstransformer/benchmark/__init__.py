from .bot_benchmark import BotBenchmark

__all__ = ["BotBenchmark"]


def __getattr__(name: str):
    """Lazy imports for heavy / CLI-oriented modules."""
    if name in ("StockfishGauntlet", "GauntletResult", "LevelResult"):
        from .engine_gauntlet import StockfishGauntlet, GauntletResult, LevelResult
        return {"StockfishGauntlet": StockfishGauntlet,
                "GauntletResult": GauntletResult,
                "LevelResult": LevelResult}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
