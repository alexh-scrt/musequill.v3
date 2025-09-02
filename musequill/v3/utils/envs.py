import re
import os

def substitute_env_vars(s: str) -> str:
    """
    Replace ${VAR} in the string with the value of the environment variable VAR.
    If the environment variable is not set, replace with an empty string.
    """

    def replacer(match: re.Match) -> str:
        var_name = match.group(1)  # Extract the part inside ${...}
        return os.getenv(var_name, "")  # Default to "" if not set

    return re.sub(r"\$\{([^}]+)\}", replacer, s)

