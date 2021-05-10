# The following modules are based on the "PyFR" implementation 
# (See licences/LICENSE_PyFR)

from configparser import SafeConfigParser, NoSectionError, NoOptionError
import io
import os
import re
import numpy as np
from dgfs1D.util import np_map
import json

cfgsect = 'config'

def _ensure_float(m):
    m = m.group(0)
    return m if any(c in m for c in '.eE') else m + '.'

class Dictionary(object):
    def __init__(self, inistr=None, defaults={}):
        self._cp = cp = SafeConfigParser(inline_comment_prefixes=[';'])

        # Preserve case
        cp.optionxform = str

        if inistr:
            cp.read_string(inistr)

        if defaults:
            cp.read_dict(defaults)

        self._dtypename = self.lookupordefault(cfgsect, 'precision', 'double')
        self._dtype = np_map[self._dtypename]
        self._dim = self.lookupordefault(cfgsect, 'dim', 1)
 
    @staticmethod
    def load(file, defaults={}):
        if isinstance(file, str):
            file = open(file)

        return Dictionary(file.read(), defaults=defaults)

    def has_section(self, section):
        return self._cp.has_section(section)

    def lookup(self, section, option, vars=None):
        val = self._cp.get(section, option, vars=vars)
        return os.path.expandvars(val)

    def lookupordefault(self, section, option, default, vars=None):
        try:
            T = type(default)
            val = T(self._cp.get(section, option, vars=vars))
        except:
            val = default
        return val

    def lookuppath(self, section, option, default, vars=None,
                abs=False):
        path = self.lookupordefault(section, option, default, vars)
        path = os.path.expanduser(path)

        if abs:
            path = os.path.abspath(path)

        return path

    def lookupexpr(self, section, option, subs={}):
        expr = self.lookup(section, option)

        # Ensure the expression does not contain invalid characters
        if not re.match(r'[A-Za-z0-9 \t\n\r.,+\-*/%()<>=\{\}\$]+$', expr):
            raise ValueError('Invalid characters in expression')

        # Substitute variables
        if subs:
            expr = re.sub(r'\b({0})\b'.format('|'.join(subs)),
                          lambda m: subs[m.group(1)], expr)

        # Convert integers to floats
        expr = re.sub(r'\b((\d+\.?\d*)|(\.\d+))([eE][+-]?\d+)?(?!\s*])',
                      _ensure_float, expr)

        # Encase in parenthesis
        return '({0})'.format(expr)

    def lookupfloat(self, section, option):
        return self._dtype(self.lookup(section, option))

    def lookupfloats(self, section, options):
        return map(lambda op: self.lookupfloat(section, op), options)

    def lookupint(self, section, option):
        return int(self.lookup(section, option))

    def lookupints(self, section, options):
        return map(lambda op: self.lookupint(section, op), options)

    def lookup_list(self, section, option, dtype):
        return np.array(list(map(
            dtype, json.loads(self.lookup(section, option)))))

    def lookupfloat_list(self, section, option):
        return self.lookup_list(section, option, self._dtype)

    def lookupint_list(self, section, option):
        return self.lookup_list(section, option, int)

    def __str__(self):
        buf = io.StringIO()
        self._cp.write(buf)
        return buf.getvalue()

    def section_values(self, section, type):
        iv = []
        for k, v in self._cp.items(section):
            try:
                try: 
                    v.index('[')
                    iv.append((k, self.lookup_list(section, k, type)))
                except ValueError:
                    iv.append((k, type(v)))
            except ValueError:
                pass
        return dict(iv)

    # Global configurations that are required in all systems
    @property
    def dtypename(self): 
        return self._dtypename

    @property
    def dtype(self): 
        return self._dtype

    @property
    def dim(self): 
        return self._dim
