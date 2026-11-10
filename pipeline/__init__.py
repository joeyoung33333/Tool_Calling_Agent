"""The agent pipeline: the Clarifier gate, the Lead, the parallel Researchers and the Writer.

Not named `agents`: that is the SDK package these modules import, and a top-level package of that
name would shadow it.

**This file must not import its own submodules.** The graph leaves the package and comes back -
`pipeline.roster` imports root `tools`, which imports `pipeline.runtime` - so importing a submodule
here makes resolution order-dependent: `import tools` then fails with `AttributeError: module
'tools' has no attribute 'ALL'` rather than an ImportError, and only on some entry points.
"""
