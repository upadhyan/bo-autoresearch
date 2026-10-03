class Refused(Exception):
    """A command's preconditions do not hold.

    The message is shown to the agent verbatim, so it says what is wrong and
    what to do instead.
    """
