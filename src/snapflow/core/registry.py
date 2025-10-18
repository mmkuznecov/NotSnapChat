"""Game registration system"""

# Simple registry using decorators
GAMES = {}


def register_game(name):
    """Decorator to register a game factory function"""

    def decorator(factory_func):
        GAMES[name] = factory_func
        return factory_func

    return decorator


def get_game(name):
    """Returns (processors_list, store) for the named game"""
    if name not in GAMES:
        raise ValueError(
            f"Game '{name}' not found. Available games: {list(GAMES.keys())}"
        )
    return GAMES[name]()
