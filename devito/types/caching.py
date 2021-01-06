import gc
import weakref

import sympy
from sympy.core import cache


__all__ = ['Cached', '_SymbolCache', 'CacheManager']

_SymbolCache = {}
"""The symbol cache."""


class AugmentedWeakRef(weakref.ref):

    def __new__(cls, obj, meta):
        obj = super().__new__(cls, obj)
        obj.nbytes = meta.get('nbytes', 0)
        return obj


class Cached(object):

    """
    Mixin class for cached symbolic objects.
    """

    @classmethod
    def _cache_key(cls, *args, **kwargs):
        """
        A unique, deterministic key from the input arguments.

        Notes
        -----
        To be implemented by subclasses.

        Returns
        -------
        The cache key. It must be hashable.
        """
        raise NotImplementedError("Subclass must implement _cache_key")

    @classmethod
    def _cache_get(cls, key):
        """
        Look up the cache for a given key.

        Parameters
        ----------
        key : object
            The cache key. It must be hashable.

        Returns
        -------
        The object if in the cache and alive, otherwise None.
        """
        if key in _SymbolCache:
            # There is indeed an object mapped to `key`. But is it still alive?
            obj = _SymbolCache[key]()
            if obj is None:
                # Cleanup _SymbolCache (though practically unnecessary)
                del _SymbolCache[key]
                return None
            else:
                return obj
        else:
            return None

    def __init__(self, key, *aliases):
        """
        Store `self` in the symbol cache.

        Parameters
        ----------
        key : object
            The cache key. It must be hashable.
        *aliases
            Additional keys to which self is mapped.
        """
        # Precompute hash. This uniquely depends on the cache key
        self._cache_key_hash = hash(key)

        # Add ourselves to the symbol cache
        awr = AugmentedWeakRef(self, self._cache_meta())
        for i in (key,) + aliases:
            _SymbolCache[i] = awr

    def __init_cached__(self, key):
        """
        Initialise `self` with a cached object state.

        Parameters
        ----------
        key : object
            The cache key of the object whose state is used to initialize `self`.
            It must be hashable.
        """
        self.__dict__ = _SymbolCache[key]().__dict__.copy()

    def __hash__(self):
        """
        The hash value of a cached object is the hash of its cache key.
        """
        return self._cache_key_hash

    def _cache_meta(self):
        """
        Metadata attached when ``self`` is added to the symbol cache.

        Notes
        -----
        This should be specialized by the individual subclasses. This is useful
        to implement callbacks to be executed upon eviction.
        """
        return {}


class CacheManager(object):

    """
    Drop unreferenced objects from the SymPy and Devito caches. The associated
    data is lost (and thus memory is freed).
    """

    gc_ths = 3*10**8
    """
    The `clear` function will trigger garbage collection if at least one weak
    reference points to an unreachable object whose size in bytes is greated
    than the `gc_ths` value. Garbage collection is an expensive operation, so
    we do it judiciously.
    """

    force_ths = 100
    """
    After `force_ths` *consecutive* calls ``clear(force=False)``, the flag
    ``force`` is ignored, and thus ``clear(force=True)`` is executed.
    ``
    """
    ncalls_w_force_false = 0

    @classmethod
    def clear(cls, force=True):
        # Wipe out the "true" SymPy cache
        cache.clear_cache()

        # Wipe out the hidden module-private SymPy caches
        sympy.polys.rootoftools.ComplexRootOf.clear_cache()
        sympy.polys.rings._ring_cache.clear()
        sympy.polys.fields._field_cache.clear()
        sympy.polys.domains.modularinteger._modular_integer_cache.clear()

        # Maybe trigger garbage collection
        if force is False:
            if cls.ncalls_w_force_false + 1 == cls.force_ths:
                # Case 1: too long since we called gc.collect, let's do it now
                gc.collect()
                cls.ncalls_w_force_false = 0
            elif any(i.nbytes > cls.gc_ths for i in _SymbolCache.values()):
                # Case 2: we got big objects in cache, we try to reclaim memory
                gc.collect()
                cls.ncalls_w_force_false = 0
            else:
                # We won't call gc.collect() this time
                cls.ncalls_w_force_false += 1
        else:
            gc.collect()

        for key, obj in list(_SymbolCache.items()):
            if obj() is None:
                del _SymbolCache[key]
