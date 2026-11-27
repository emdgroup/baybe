"""Finite-reference partial knowledge gradient (pKGFN) for multi-step networks.

The whole pKGFN path lives in this package. Read it in this order:

1. :mod:`baybe.multistep.kgfn.gain` -- the unweighted gain
   ``max(0, mean_f(max_r E[U] - baseline))`` over a finite reference set.
2. :mod:`baybe.multistep.kgfn.references` -- how that finite reference set is
   sampled without building a parameter product.
3. :mod:`baybe.multistep.kgfn.direct` -- the executable local action, the local
   success probability ``p``, the cost ``C``, and the selection score
   ``KG(a) / C(a)`` with ``KG(a) = p(a) * gain(a)``.

The package deliberately imports no submodule eagerly: ``acquisition`` imports
``kgfn.gain``, so an eager import of ``kgfn.direct`` here would re-enter a
partially initialized ``acquisition`` module.
"""
