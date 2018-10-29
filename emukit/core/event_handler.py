class EventHandler(list):
    """
    A list of callable objects. Calling an instance of this will cause a call to each item in the list in ascending
    order by index.
    """
    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)