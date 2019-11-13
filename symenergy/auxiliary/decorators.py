#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""
import wrapt
from hashlib import md5
from symenergy import _get_logger

logger = _get_logger('decorators')

@wrapt.decorator
def hexdigest(f, self, *args, **kwargs):

    name = self.name if hasattr(self, 'name') else 'unnamed'

    hash_input = f()
    hash_ = md5(hash_input.encode('utf-8')).hexdigest()

    logger.debug('Generated hash %s for object %s.' % (hash_, name))

    return hash_
