# Copyright 2024 The Emukit Authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


class EventHandler(list):
    """
    A list of callable objects. Calling an instance of this will cause a call to each item in the list in ascending
    order by index.

    Code taken from: https://stackoverflow.com/a/2022629

    To subscribe to the event simply append a function to the event handler:
    ``event_handler.append(fcn_to_call_on_event)``
    """

    def __call__(self, *args, **kwargs):
        for f in self:
            f(*args, **kwargs)

    def __repr__(self):
        return "Event(%s)" % list.__repr__(self)
